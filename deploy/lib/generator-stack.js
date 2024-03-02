import * as path from 'path';
import {
  Stack,
  aws_iam as iam,
  Duration,
  CfnOutput,
} from 'aws-cdk-lib';
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { DockerImageFunction,DockerImageCode,Architecture }  from 'aws-cdk-lib/aws-lambda';
import {ApiGatewayStack} from './apigw-stack.js';
import { fileURLToPath } from "url";
import { join } from "path";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class GeneratorCdkStack extends Stack {
  /**
   *
   * @param {Construct} scope
   * @param {string} id
   * @param {StackProps=} props
   */
  constructor(scope, id, props) {
    super(scope, id, props);

    const region = props.env.region;
    const account_id = Stack.of(this).account;

    const existed_vpc = ec2.Vpc.fromLookup(this, 'chatbot-deploy-vpc', {
      vpcName: 'QAChatDeployStack/vpc-stack/QAChat-workshop-Vpc',
    });

    const securityGroup = new ec2.SecurityGroup(this,'lambda-security-group',{
      vpc: existed_vpc, description: 'sg_for_generater_lambda'
    });

    securityGroup.addIngressRule(securityGroup, ec2.Port.allTraffic(), 'Allow self traffic');

    const generator_lambda = new DockerImageFunction(this,
      "generator_lambda", {
      code: DockerImageCode.fromImageAsset(join(__dirname, "../../code/generator")),
      timeout: Duration.minutes(15),
      memorySize: 1024,
      runtime: 'python3.9',
      functionName: 'Invoke_Generator',
      vpc:existed_vpc,
      vpcSubnets:existed_vpc.privateSubnets,
      securityGroups:[securityGroup],
      architecture: Architecture.X86_64,
      environment: {
        other_accounts:'',
        region:region
      },
    });

    // Grant the Lambda function can invoke sagemaker
    generator_lambda.addToRolePolicy(new iam.PolicyStatement({
      // principals: [new iam.AnyPrincipal()],
        actions: [ 
          "sagemaker:InvokeEndpointAsync",
          "sagemaker:InvokeEndpoint",
          "dynamodb:*",
          "secretsmanager:GetSecretValue",
          "ssm:GetParameters",
          "ssm:GetParameter",
          "bedrock:*"
          ],
        effect: iam.Effect.ALLOW,
        resources: ['*'],
        }))

    //create REST api
    const generator_restapi = new ApiGatewayStack(this,'GeneratorRestApi',{lambda_fn:generator_lambda,name:"generator_entry"})

    new CfnOutput(this, 'API gateway endpoint url', {value:`${generator_restapi.endpoint}`});
    new CfnOutput(this, 'existed_vpc_id', {value:existed_vpc.vpcId});
  }
}

