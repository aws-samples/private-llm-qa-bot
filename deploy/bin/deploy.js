#!/usr/bin/env node
import cdk  from 'aws-cdk-lib';
import { DeployStack } from '../lib/deploy-stack.js';
import { FrontendCdkStack } from '../lib/frontend-cdk-stack.js';
import { BedrockCdkStack } from '../lib/bedrock-stack.js';
import { GeneratorCdkStack } from '../lib/generator-stack.js';
// import { exec } from 'child_process';
import * as dotenv from 'dotenv' ;
dotenv.config()

console.log(process.env.CDK_DEFAULT_ACCOUNT,process.env.CDK_DEFAULT_REGION);
const app = new cdk.App();
new DeployStack(app, 'QAChatDeployStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION},
});

new FrontendCdkStack(app, 'ChatFrontendDeployStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});

// you need to install by `npm i bedrock-agents-cdk`
new BedrockCdkStack(app, 'BedrockKBDeployStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION, s3bucket: process.env.UPLOAD_BUCKET, s3prefix : process.env.UPLOAD_OBJ_PREFIX },
});

new GeneratorCdkStack(app, 'GeneratorStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION},
});

// const frontEndScriptPath = '../chatbotFE/gen_env.sh';

// const command = process.argv[2];
// if (command === 'deploy'){
//   exec(`bash ${frontEndScriptPath} ${process.env.CDK_DEFAULT_REGION }`, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`Error executing script: ${error}`);
//       return;
//     }
  
//     console.log(`Script output: ${stdout}`);
//     console.error(`Script errors: ${stderr}`);
//   });
// }

