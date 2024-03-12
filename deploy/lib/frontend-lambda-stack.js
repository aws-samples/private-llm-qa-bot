import { NestedStack, Duration, CfnOutput } from "aws-cdk-lib";
import {
  LambdaIntegration,
  RestApi,
  TokenAuthorizer,
  Cors,
  ResponseType,
  EndpointType
} from "aws-cdk-lib/aws-apigateway";
import * as s3 from "aws-cdk-lib/aws-s3";

import * as sns from "aws-cdk-lib/aws-sns";
import subscriptions from "aws-cdk-lib/aws-sns-subscriptions";
import lambda from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { join } from "path";
import * as dotenv from "dotenv";
import { WebSocketLambdaIntegration } from "@aws-cdk/aws-apigatewayv2-integrations-alpha";
import * as apigwv2 from "@aws-cdk/aws-apigatewayv2-alpha";
import * as ecr from "aws-cdk-lib/aws-ecr";


dotenv.config();

export class LambdaStack extends NestedStack {
  handlersMap;
  apigw_url;
  login_fn;
  auth_fn;
  users_fn;
  job_fn;
  update_job_fn;
  topicArn;
  lambda_connect_handle;
  lambda_handle_chat;
  lambda_list_idx;
  lambda_handle_upload;
  webSocketURL;

  /**
   *
   * @param {Construct} scope
   * @param {string} id
   * @param {StackProps=} props
   */
  constructor(scope, id, props) {
    super(scope, id, props);

    const user_table = props.user_table;
    const agents_table = props.agents_table;
    const prompt_hub_table = props.prompt_hub_table;
    const model_hub_table = props.model_hub_table;
    this.handlersMap = new Map();

    const createNodeJsLambdaFn = (scope, path, index_fname, api, envProps) => {
      let handler = new NodejsFunction(scope, api, {
        entry: join(path, index_fname),
        depsLockFilePath: join(path, "package-lock.json"),
        ...envProps,
      });
      this.handlersMap.set(api, handler);
      return handler;
    };

    // Create sns Topic
    const snsTopic = new sns.Topic(this, "Topic", {
        displayName: "chat messages topic",
        });
      
    this.topicArn = snsTopic.topicArn;


    const commonProps = {
      bundling: {
        externalModules: ["@aws-sdk"],
      },
      environment: {
        USER_TABLE_NAME: user_table.tableName,
        TOKEN_KEY: process.env.TOKEN_KEY,
        SNS_TOPIC_ARN: snsTopic.topicArn,
        UPLOAD_BUCKET: process.env.UPLOAD_BUCKET,
        UPLOAD_OBJ_PREFIX:process.env.UPLOAD_OBJ_PREFIX,
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        START_CMD: process.env.START_CMD,
      },
      runtime: lambda.Runtime.NODEJS_18_X,
      memorySize: 256,
      timeout: Duration.minutes(1),
    };

    this.login_fn = createNodeJsLambdaFn(
      this,
      "lambda/login",
      "index.js",
      "login",
      {
        ...commonProps,
        bundling: {
          externalModules: ["@aws-sdk"],
          nodeModules: ["jsonwebtoken", "bcryptjs"],
        },
      }
    );
    user_table.grantReadWriteData(this.login_fn);

    this.auth_fn = createNodeJsLambdaFn(
      this,
      "lambda/auth",
      "index.js",
      "lambda_auth",
      {
        ...commonProps,
        bundling: {
          externalModules: ["@aws-sdk"],
          nodeModules: ["jsonwebtoken"],
        },
      }
    );
    user_table.grantReadWriteData(this.auth_fn);

    this.users_fn = createNodeJsLambdaFn(
      this,
      "lambda/admin_users",
      "index.js",
      "users",
      {
        ...commonProps,
        bundling: {
          externalModules: ["@aws-sdk"],
          nodeModules: ["bcryptjs"],
        },
      }
    );
    user_table.grantReadWriteData(this.users_fn);
    
    const layer = new lambda.LayerVersion(this, 'ChatbotLayer', {
      code: lambda.Code.fromAsset('layer/ChatbotFELayer.zip'),
      description: 'ChatbotFELayer Python helper utility',
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_9],
      layerVersionName:'ChatbotFELayer',
    });


    this.lambda_chat_py = new lambda.Function(this, 'handle_chat_py',{
      code: lambda.Code.fromAsset('lambda/lambda_chat_py'),
      layers:[layer],
      handler: 'app.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      timeout: Duration.minutes(5),
      environment: {
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        MAIN_FUN_ARN:process.env.MAIN_FUN_ARN,
        all_in_one_api:process.env.all_in_one_api,
        sd_endpoint_name:process.env.sd_endpoint_name
      },
      memorySize: 256,
    })

    //read sd image data from s3 
    const sgbucket = s3.Bucket.fromBucketAttributes(this,'sagemakerbucket',{bucketName:`sagemaker-${process.env.CDK_DEFAULT_REGION}-${process.env.CDK_DEFAULT_ACCOUNT}`});
    sgbucket.grantRead(this.lambda_chat_py);


    this.lambda_connect_handle = createNodeJsLambdaFn(
      this,
      "lambda/lambda_connect_handle",
      "index.mjs",
      "lambda_connect_handle",
      {
        ...commonProps,
        bundling: {
          externalModules: ["@aws-sdk"],
          nodeModules: ["jsonwebtoken"],
        },
      }
    );

    this.lambda_handle_chat = createNodeJsLambdaFn(
      this,
      "lambda/lambda_handle_chat",
      "index.mjs",
      "lambda_handle_chat",
      {
        ...commonProps,
      }
    );

    this.lambda_list_idx = createNodeJsLambdaFn(
      this,
      "lambda/lambda_list_idx",
      "index.mjs",
      "lambda_list_idx",
      {
        ...commonProps,
        environment: {
          AGENTS_TABLE_NAME:agents_table.tableName,
          MAIN_FUN_ARN:process.env.MAIN_FUN_ARN
        },
      }
    );
  
    this.lambda_handle_upload = createNodeJsLambdaFn(
      this,
      "lambda/lambda_handle_upload",
      "index.js",
      "lambda_handle_upload",
      {
        ...commonProps,
        timeout: Duration.minutes(5),
        memorySize: 512,
        bundling: {
          externalModules: ["@aws-sdk"],
          nodeModules: ["formidable","busboy"],
        },
      }
    );
    
    // prompt hub 管理函数
    this.lambda_prompt_hub = new lambda.Function(this, 'lambda_prompthub',{
      code: lambda.Code.fromAsset('lambda/lambda_prompthub'),
      handler: 'app.handler',
      runtime: lambda.Runtime.PYTHON_3_10,
      timeout: Duration.minutes(3),
      environment: {
      },
      memorySize: 256,
    })
    prompt_hub_table.grantReadWriteData(this.lambda_prompt_hub);

    // model hub 管理函数
    this.lambda_model_hub = new lambda.Function(this, 'lambda_modelhub',{
      code: lambda.Code.fromAsset('lambda/lambda_modelhub'),
      handler: 'app.handler',
      runtime: lambda.Runtime.PYTHON_3_10,
      timeout: Duration.minutes(3),
      environment: {
      },
      memorySize: 256,
    })
    model_hub_table.grantReadWriteData(this.lambda_model_hub);


    // doc_index_table.grantReadWriteData(this.lambda_list_idx )
    const bucket = s3.Bucket.fromBucketName(this, 'DocUploadBucket',process.env.UPLOAD_BUCKET);
    bucket.grantReadWrite(this.lambda_handle_upload);

    const main_fn = lambda.Function.fromFunctionArn(this,'main func',process.env.MAIN_FUN_ARN);
    main_fn.grantInvoke(this.lambda_chat_py);
    main_fn.grantInvoke(this.lambda_list_idx);

    agents_table.grantReadWriteData(this.lambda_list_idx);

    const api = new RestApi(this, "ChatbotFERestApi", {
      cloudWatchRole: true,
      defaultCorsPreflightOptions: {
        allowOrigins: Cors.ALL_ORIGINS,
        allowHeaders: Cors.DEFAULT_HEADERS,
        allowMethods: Cors.ALL_METHODS,
      },
      endpointConfiguration:{types:[EndpointType.REGIONAL]}

    });

    api.addGatewayResponse("cors1", {
      type: ResponseType.ACCESS_DENIED,
      statusCode: "500",
      responseHeaders: {
        "Access-Control-Allow-Origin": "'*'",
      },
    });
    api.addGatewayResponse("cors2", {
      type: ResponseType.DEFAULT_4XX,
      statusCode: "400",
      responseHeaders: {
        "Access-Control-Allow-Origin": "'*'",
      },
    });
    api.addGatewayResponse("cors3", {
      type: ResponseType.DEFAULT_5XX,
      statusCode: "500",
      responseHeaders: {
        "Access-Control-Allow-Origin": "'*'",
      },
    });

    this.apigw_url = api.url;

    //create lambda authorizer
    const authorizerFn = this.auth_fn;
    const authorizer = new TokenAuthorizer(this, "APIAuthorizer", {
      handler: authorizerFn,
      resultsCacheTtl: Duration.minutes(0),
    });

    const uploadIntegration = new LambdaIntegration(this.lambda_handle_upload);
    const upload = api.root.addResource('upload');
    upload.addMethod('POST', uploadIntegration,{authorizer});

    const docsIntegration = new LambdaIntegration(this.lambda_list_idx );
    const docs = api.root.addResource('docs');
    docs.addMethod('GET',docsIntegration,{authorizer});
    docs.addMethod('DELETE',docsIntegration,{authorizer});

    const templateIntegration = new LambdaIntegration(this.lambda_list_idx );
    const template = api.root.addResource('template');
    template.addMethod('GET',templateIntegration,{authorizer});
    template.addMethod('POST',templateIntegration,{authorizer});
    template.addMethod('DELETE',templateIntegration,{authorizer});

    const feedbackIntegration = new LambdaIntegration(this.lambda_list_idx );
    const feedback = api.root.addResource('feedback');
    feedback.addMethod('POST',feedbackIntegration,{authorizer});
    feedback.addMethod('DELETE',feedbackIntegration,{authorizer});
    feedback.addMethod('GET',feedbackIntegration,{authorizer});

    const agentsIntegration = new LambdaIntegration(this.lambda_list_idx );
    const agents = api.root.addResource('agents');
    agents.addMethod('POST',agentsIntegration,{authorizer});
    agents.addMethod('DELETE',agentsIntegration,{authorizer});
    agents.addMethod('GET',agentsIntegration,{authorizer});
    agents.addResource("{id}").addMethod("GET",agentsIntegration,{authorizer});


     // prompt hub api
    const promptHubIntegration = new LambdaIntegration(this.lambda_prompt_hub );
    const prompt_hub = api.root.addResource('prompt_hub');
    prompt_hub.addMethod('POST',promptHubIntegration,{authorizer});
    prompt_hub.addMethod('DELETE',promptHubIntegration,{authorizer});
    prompt_hub.addMethod('GET',promptHubIntegration,{authorizer});
    prompt_hub.addResource("{id}").addMethod("GET",promptHubIntegration,{authorizer});

     // model hub api
     const modelHubIntegration = new LambdaIntegration(this.lambda_model_hub );
     const model_hub = api.root.addResource('model_hub');
     model_hub.addMethod('POST',modelHubIntegration,{authorizer});
     model_hub.addMethod('DELETE',modelHubIntegration,{authorizer});
     model_hub.addMethod('GET',modelHubIntegration,{authorizer});
     model_hub.addResource("{id}").addMethod("GET",modelHubIntegration,{authorizer});

    const loginIntegration = new LambdaIntegration(this.login_fn);
    const login = api.root.addResource("login");
    login.addMethod("POST", loginIntegration);

    const adminUsersIntegration = new LambdaIntegration(this.users_fn);
    const users = api.root.addResource("users");
    users.addMethod("GET", adminUsersIntegration, { authorizer });
    users.addMethod("POST", adminUsersIntegration, { authorizer });

    const singleUser = users.addResource("{id}");
    singleUser.addMethod("GET", adminUsersIntegration, { authorizer });

    // create websocket apigw
    const webSocketApi = new apigwv2.WebSocketApi(this, "ChatBotWsApi", {
      connectRouteOptions: {
        integration: new WebSocketLambdaIntegration(
          "ConnectIntegration",
          this.lambda_connect_handle
        ),
      },
    });

    const webSocketStage = new apigwv2.WebSocketStage(this, "mystage", {
      webSocketApi,
      stageName: "Prod",
      autoDeploy: true,
    });
    this.webSocketURL = webSocketStage.url;

    webSocketApi.addRoute("sendprompt", {
      integration: new WebSocketLambdaIntegration(
        "SendMessageIntegration",
        this.lambda_handle_chat
      ),
    });

    // per stage permission
    webSocketStage.grantManagementApiAccess(this.lambda_chat_py);

    // for all the stages permission
    webSocketApi.grantManageConnections(this.lambda_chat_py);


      //Add the lambda subscription
      snsTopic.addSubscription(new subscriptions.LambdaSubscription(this.lambda_chat_py));
      // Grant the Lambda function publish data
      snsTopic.grantPublish(this.lambda_handle_chat);
  }
}
