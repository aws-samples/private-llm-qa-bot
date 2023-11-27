#!/usr/bin/env node
import cdk  from 'aws-cdk-lib';
import { DeployStack } from '../lib/deploy-stack.js';
import { FrontendCdkStack } from '../lib/frontend-cdk-stack.js';
import * as dotenv from 'dotenv' ;
dotenv.config()

console.log(process.env.CDK_DEFAULT_ACCOUNT,process.env.CDK_DEFAULT_REGION);
const app = new cdk.App();
new DeployStack(app, 'QAChatDeployStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});

new FrontendCdkStack(app, 'ChatFrontendDeployStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
