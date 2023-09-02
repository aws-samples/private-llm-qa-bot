import { Stack, CfnOutput } from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as sagemaker from "aws-cdk-lib/aws-sagemaker"

export class SagemakerDomainStack extends Stack {
  constructor(scope, id, props) {
    super(scope, id, props);

    const region = props.env.region;
    const account_id = Stack.of(this).account;
    
    const defaultVpc = ec2.Vpc.fromLookup(this, 'DefaultVPC', { isDefault: true });
    const subnetIds = [];
    
    defaultVpc.publicSubnets.forEach((subnet, index) => {
      subnetIds.push(subnet.subnetId);
    });
    
    const sagemakerExecutionRole = new iam.Role(this, 'sagemaker-execution-role', {
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal('glue.amazonaws.com'), new iam.ServicePrincipal('sagemaker.amazonaws.com')),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess')
      ],
      roleName: "sagemaker_execute_role"
    });

    // Create the SageMaker Domain
    const domain = new sagemaker.CfnDomain(this, 'SageMakerDomain', {
      domainName: 'chatbot-sagemaker-domain', // Replace with your preferred domain name
      authMode: 'IAM',
      vpcId: defaultVpc.vpcId, // Replace with your VPC ID
      subnetIds: subnetIds, // Replace with your subnet IDs
      defaultUserSettings: {
        executionRole: sagemakerExecutionRole.roleArn,
      },
    });

    // SageMaker User Profile
    const userProfile = new sagemaker.CfnUserProfile(this, 'LLMUser', {
      domainId: domain.attrDomainId,
      userProfileName: 'llm-user-1',
      userSettings: {
        executionRole: sagemakerExecutionRole.roleArn,
      },
    })
    
    new CfnOutput(this,'SagemakerARN',{value:sagemakerExecutionRole.roleArn});
  }
}