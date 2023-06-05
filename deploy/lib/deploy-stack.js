// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import { Stack, Duration, CfnOutput, RemovalPolicy } from "aws-cdk-lib";
import { DockerImageFunction }  from 'aws-cdk-lib/aws-lambda';
import { DockerImageCode,Architecture } from 'aws-cdk-lib/aws-lambda';
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as iam from "aws-cdk-lib/aws-iam";
import { AttributeType, Table } from "aws-cdk-lib/aws-dynamodb";
import * as ecr from 'aws-cdk-lib/aws-ecr';
import { VpcStack } from './vpc-stack.js';
import {GlueStack} from './glue-stack.js';
import {OpenSearchStack} from './opensearch-stack.js';
import {ApiGatewayStack} from './apigw-stack.js';
// import { ALBStack } from "./alb-stack.js";
import { Ec2Stack } from "./ec2-stack.js";
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3n from 'aws-cdk-lib/aws-s3-notifications';
import * as dotenv from "dotenv";
import * as ec2 from "aws-cdk-lib/aws-ec2";

dotenv.config();

import path from "path";
import { fileURLToPath } from "url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import { join } from "path";
export class DeployStack extends Stack {
  /**
   *
   * @param {Construct} scope
   * @param {string} id
   * @param {StackProps=} props
   */
  constructor(scope, id, props) {
    super(scope, id, props);

  
    // const region = process.env.CDK_DEFAULT_REGION;
    // const account = process.env.CDK_DEFAULT_ACCOUNT;
    const region = props.env.region;
    const account_id = Stack.of(this).account;
    const aos_existing_endpoint = props.env.aos_existing_endpoint;

    const vpcStack = new VpcStack(this,'vpc-stack',{env:process.env});
    const vpc = vpcStack.vpc;
    const subnets = vpcStack.subnets;
    const securityGroups = vpcStack.securityGroups;

    const cn_region = ["cn-north-1","cn-northwest-1"];

    
    if (!cn_region.includes(region)) {
      const ec2stack = new Ec2Stack(this,'Ec2Stack',{vpc:vpc,securityGroup:securityGroups[0]});
      new CfnOutput(this, 'OpenSearch EC2 Proxy Address', { value: `http://${ec2stack.publicIP}:8081/_dashboards/`});
      new CfnOutput(this, 'Download Key Command', { value: 'aws secretsmanager get-secret-value --secret-id ec2-ssh-key/cdk-keypair/private --query SecretString --output text > cdk-key.pem && chmod 400 cdk-key.pem' })
      new CfnOutput(this, 'ssh command', { value: 'ssh -i cdk-key.pem -o IdentitiesOnly=yes ec2-user@' + ec2stack.dnsName})
      ec2stack.addDependency(vpcStack);
    }

      // Create open search if the aos endpoint not provided
    let opensearch_endpoint=aos_existing_endpoint;
    let opensearchStack;
    if (!aos_existing_endpoint || aos_existing_endpoint === 'optional'){
         opensearchStack = new OpenSearchStack(this,'os-chat-dev',
              {vpc:vpc,subnets:subnets,securityGroup:securityGroups[0]});
        opensearch_endpoint = opensearchStack.domainEndpoint;
        opensearchStack.addDependency(vpcStack);
    }
    new CfnOutput(this,'opensearch endpoint',{value:opensearch_endpoint});
    new CfnOutput(this,'region',{value:process.env.CDK_DEFAULT_REGION});
    new CfnOutput(this,'UPLOAD_BUCKET',{value:process.env.UPLOAD_BUCKET});
    new CfnOutput(this,'llm_chatglm_endpoint',{value:process.env.llm_chatglm_endpoint});
    new CfnOutput(this,'embedding_endpoint',{value:process.env.embedding_endpoint});
    new CfnOutput(this,'model_name',{value:process.env.llm_chatglm_endpoint.replace('-endpoint','')});
    new CfnOutput(this,'embedding_model_name',{value:process.env.embedding_endpoint.replace('-endpoint','')});


    


    // allow the ec2 sg traffic  
    // securityGroups[0].addIngressRule(ec2stack.securityGroup, ec2.Port.allTraffic(), 'Allow SSH Access')


    // const albstack = new ALBStack(this,'ALBstack',{vpc:vpc,instanceId:ec2stack.instanceId});
    // new CfnOutput(this,'ALB dnsname',{value:albstack.dnsName});



    const chat_session_table = new Table(this, "chatbot_session_info", {
      partitionKey: {
        name: "session-id",
        type: AttributeType.STRING,
      },
      removalPolicy: RemovalPolicy.DESTROY, // NOT recommended for production code
    });

    const lambda_main_brain = new DockerImageFunction(this,
      "lambda_main_brain", {
      code: DockerImageCode.fromImageAsset(join(__dirname, "../lambda/main_brain")),
      timeout: Duration.minutes(15),
      memorySize: 1024,
      runtime: 'python3.9',
      // functionName: 'Main_brain',
      vpc:vpc,
      vpcSubnets:subnets,
      securityGroups:securityGroups,
      architecture: Architecture.X86_64,
      environment: {
        aos_endpoint:opensearch_endpoint,
        Kendra_index_id:process.env.Kendra_index_id ,
        Kendra_result_num:process.env.Kendra_result_num ,
        aos_index:process.env.aos_index ,
        aos_knn_field:process.env.aos_knn_field ,
        aos_results:process.env.aos_results ,
        embedding_endpoint:process.env.embedding_endpoint ,
        llm_default_endpoint:process.env.llm_default_endpoint,
        llm_bloomz_endpoint:process.env.llm_bloomz_endpoint,
        llm_chatglm_endpoint:process.env.llm_chatglm_endpoint,
        chat_session_table:chat_session_table.tableName,
      },
    });

    // Grant the Lambda function can invoke sagemaker
    lambda_main_brain.addToRolePolicy(new iam.PolicyStatement({
      // principals: [new iam.AnyPrincipal()],
        actions: [ 
          "sagemaker:InvokeEndpointAsync",
          "sagemaker:InvokeEndpoint",
          "s3:List*",
          "s3:Put*",
          "s3:Get*",
          "es:*",
          ],
        effect: iam.Effect.ALLOW,
        resources: ['*'],
        }))
      
    chat_session_table.grantReadWriteData(lambda_main_brain);


    //glue job
    const gluestack = new GlueStack(this,'glue-stack',{opensearch_endpoint,region,vpc,subnets,securityGroups});
    new CfnOutput(this, `Glue Job name`,{value:`${gluestack.jobName}`});
    gluestack.addDependency(vpcStack)

    //file upload bucket
    const bucket = new s3.Bucket(this, 'DocUploadBucket', {
      removalPolicy: RemovalPolicy.DESTROY,
      bucketName:process.env.UPLOAD_BUCKET
    });

    const layer = new lambda.LayerVersion(this, 'ChatbotLayer', {
      code: lambda.Code.fromAsset(path.join(__dirname,'../../code/layer_asset')),
      description: 'ChatbotLayer Python helper utility',
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_8],
      removalPolicy: RemovalPolicy.DESTROY,
      layerVersionName:'ChatbotLayer',
    });

    const offline_trigger_lambda =  new lambda.Function(this, 'offline_trigger_lambda', {
          environment: {
            glue_jobname:gluestack.jobName,
            embedding_endpoint:process.env.embedding_endpoint
          },
          runtime: lambda.Runtime.PYTHON_3_9,
          timeout: Duration.minutes(1),
          // functionName:'offline_trigger_lambda',
          handler: 'offline_trigger_lambda.lambda_handler',
          code: lambda.Code.fromAsset(path.join(__dirname,'../../code/lambda_offline_trigger')),
          vpc:vpc,
          vpcSubnets:subnets,
        });

    // offline_trigger_lambda.role?.addManagedPolicy('arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole')
    offline_trigger_lambda.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['s3:GetBucketNotification', 's3:PutBucketNotification'],
        effect: iam.Effect.ALLOW,
        resources: [ bucket.bucketArn ]
      })
    );
    offline_trigger_lambda.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['glue:StartJobRun'],
        effect: iam.Effect.ALLOW,
        resources: [ gluestack.jobArn ]
      })
    );

    bucket.addEventNotification(
      s3.EventType.OBJECT_CREATED_PUT,
      new s3n.LambdaDestination(offline_trigger_lambda),{
          prefix: process.env.UPLOAD_OBJ_PREFIX,
      }
  )

    bucket.addEventNotification(
      s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD,
      new s3n.LambdaDestination(offline_trigger_lambda),{
          prefix: process.env.UPLOAD_OBJ_PREFIX,
      }
  )

       //create REST api
    const restapi = new ApiGatewayStack(this,'ChatBotRestApi',{lambda_fn:lambda_main_brain})
    new CfnOutput(this, `API gateway endpoint url`,{value:`${restapi.endpoint}`});

    const role = new iam.Role(this, 'chatbot-kinesis-firehose', {
      assumedBy: new iam.ServicePrincipal('logs.amazonaws.com'),
    });
    new CfnOutput(this, 'Kinesis_Firehose_Role',{value:`${role.roleName}`});

    const logResource =  (region.startsWith('cn'))?
                `arn:aws-cn:logs:${region}:${account_id}:log-group:*`:
                `arn:aws:logs:${region}:${account_id}:log-group:*`;

    const policy = new iam.Policy(this, 'chatbot-kinesis-policy', {
      statements: [
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            'kinesis:PutRecord',
            'firehose:PutRecord',
            'kinesis:PutRecords',
            'firehose:PutRecordBatch',
          ],
          resources: ['*'],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            's3:PutObject',
            's3:GetObject',
            's3:ListBucketMultipartUploads',
            's3:AbortMultipartUpload',
            's3:ListBucket',
            'logs:PutLogEvents',
            's3:GetBucketLocation',
          ],
          resources: [
            logResource,
            `${bucket.bucketArn}/*`,
            `${bucket.bucketArn}`,
          ],
        }),
      ],
    });
    
    role.attachInlinePolicy(policy);
  }
}
