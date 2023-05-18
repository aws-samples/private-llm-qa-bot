import { App, Stack, StackProps, Duration, CfnOutput } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as aws_ec2 from 'aws-cdk-lib/aws-ec2';
import * as aws_os from 'aws-cdk-lib/aws-opensearchservice';
import * as aws_lambda from 'aws-cdk-lib/aws-lambda';
import * as aws_api_gateway from 'aws-cdk-lib/aws-apigateway'

import * as path from 'path';

export class chatbotMain extends Stack {
  constructor(scope: Construct, id: string, props: StackProps = {}) {
    super(scope, id, props);

    // cfn parameters to specific vpc id
    // new CfnParameter(this, 'VpcId', {
    //   type: 'AWS::EC2::VPC::Id',
    //   description: 'VPC ID',
    //   default: 'vpc-0a0a0a0a0a0a0a0a0',
    // });

    // search for existing vpc with filter as default or vpc id
    const vpc = aws_ec2.Vpc.fromLookup(this, 'VPC', {
      isDefault: true,
      // vpcId: vpcId.valueAsString,
    });

    // create opensearch domain
    const openSearchDomain = new aws_os.Domain(this, 'OpenSearchDomain', {
      version: aws_os.EngineVersion.OPENSEARCH_2_5,
      vpc,
      // specified public vpc subnets
      vpcSubnets: [
        {
          // subnetType: aws_ec2.SubnetType.PUBLIC,
          // pick the first public subnet
          subnets: [vpc.selectSubnets({ subnetType: aws_ec2.SubnetType.PUBLIC }).subnets[0]],
        },
      ],
      capacity: {
        dataNodeInstanceType: 't3.small.search',
        dataNodes: 1,
      },
    });
    
    // lambda to index data to opensearch, implemented by lambda container image
    const indexLambda = new aws_lambda.DockerImageFunction(this, 'IndexLambda', {
      code: aws_lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../lambda/index')),
      memorySize: 1024,
      timeout: Duration.minutes(5),
      environment: {
        OPENSEARCH_HOST: openSearchDomain.domainEndpoint,
        OPENSEARCH_PORT: '443',
      },
      vpc: vpc,
      vpcSubnets: {
        subnets: [vpc.selectSubnets({ subnetType: aws_ec2.SubnetType.PUBLIC }).subnets[0]],
      },
      allowPublicSubnet: true,
    });

    // lambda to search data from opensearch, implemented by lambda container image
    const searchLambda = new aws_lambda.DockerImageFunction(this, 'SearchLambda', {
      code: aws_lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../lambda/search')),
      memorySize: 1024,
      timeout: Duration.minutes(5),
      environment: {
        OPENSEARCH_HOST: openSearchDomain.domainEndpoint,
        OPENSEARCH_PORT: '443',
      },
      vpc: vpc,
      vpcSubnets: {
        subnets: [vpc.selectSubnets({ subnetType: aws_ec2.SubnetType.PUBLIC }).subnets[0]],
      },
      allowPublicSubnet: true,
    });

    // grant lambda to access opensearch domain
    openSearchDomain.grantReadWrite(indexLambda);
    openSearchDomain.grantRead(searchLambda);

    // api gateway to expose lambda function
    const chatbotAPI = new aws_api_gateway.RestApi(this, 'chatbot', {
      restApiName: 'chatbot-api',
      description: 'This service serves chatbot api.',
      endpointConfiguration: {
        types: [aws_api_gateway.EndpointType.REGIONAL],
      },
    });

    // api gateway integration with lambda
    const indexIntegration = new aws_api_gateway.LambdaIntegration(indexLambda);
    const searchIntegration = new aws_api_gateway.LambdaIntegration(searchLambda);

    // api gateway resource
    const indexResource = chatbotAPI.root.addResource('index');
    const searchResource = chatbotAPI.root.addResource('search');

    // api gateway method
    indexResource.addMethod('POST', indexIntegration);
    searchResource.addMethod('POST', searchIntegration);

    // output opensearch endpoint and api gateway endpoint with cfn output
    new CfnOutput(this, 'OpenSearchEndpoint', {
      value: openSearchDomain.domainEndpoint,
    });
    
    new CfnOutput(this, 'APIGatewayEndpoint', {
      value: chatbotAPI.url,
    });

  }
}

// for development, use account/region from cdk cli
const devEnv = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: "ap-northeast-1", //process.env.CDK_DEFAULT_REGION,
};

const app = new App();

new chatbotMain(app, 'chatbot-dev', { env: devEnv });

app.synth();