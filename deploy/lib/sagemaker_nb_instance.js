import { Stack, CfnOutput } from "aws-cdk-lib";
import * as cdk from 'aws-cdk-lib';
import * as iam from "aws-cdk-lib/aws-iam";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as sagemaker from "aws-cdk-lib/aws-sagemaker"

export class SagemakerNotebookStack extends Stack {
  constructor(scope, id, props) {
    super(scope, id, props);

    const region = props.env.region;
    const account_id = Stack.of(this).account;
    
    const sagemakerExecutionRole = new iam.Role(this, 'sagemaker-execution-role', {
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal('glue.amazonaws.com'), new iam.ServicePrincipal('sagemaker.amazonaws.com')),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess')
      ],
      roleName: "sagemaker_execute_role"
    });
    
    const lifecycleCode = [
        {"content": cdk.Fn.base64(`
        cd /home/ec2-user/SageMaker/
        curl -O https://aws-jam-challenge-resources.s3.amazonaws.com/e-2-e-genai-abandon-loneliness/JAM-Task.ipynb -O https://aws-jam-challenge-resources.s3.amazonaws.com/e-2-e-genai-abandon-loneliness/image-1.png -O https://aws-jam-challenge-resources.s3.amazonaws.com/e-2-e-genai-abandon-loneliness/image-2.png
        `) }
    ];
    
    const sageMakerIntanceLifecyclePolicy = new sagemaker.CfnNotebookInstanceLifecycleConfig(this, 'notebookLifecyclePolicy', {
        notebookInstanceLifecycleConfigName: "pull-notebook-for-jam",
        onCreate: lifecycleCode,
        onStart: lifecycleCode
    });
    
    const notebookInstance = new sagemaker.CfnNotebookInstance(this, 'JAM-NotebookInstance', {
      instanceType: 'ml.g4dn.2xlarge', 
      roleArn: sagemakerExecutionRole.roleArn, 
      volumeSizeInGb: 40, 
      directInternetAccess: 'Enabled', 
      lifecycleConfigName: sageMakerIntanceLifecyclePolicy.notebookInstanceLifecycleConfigName
    });
    

    new CfnOutput(this,'SagemakerARN',{value:sagemakerExecutionRole.roleArn});
  }
}