
import { NestedStack, CfnOutput,RemovalPolicy }  from 'aws-cdk-lib';
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Asset } from 'aws-cdk-lib/aws-s3-assets';
import path from "path";
import { fileURLToPath } from "url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class Ec2Stack extends NestedStack {

    instanceId = '';
    dnsName = '';
    /**
     *
     * @param {Construct} scope
     * @param {string} id
     * @param {StackProps=} props
     */
    constructor(scope, id, props) {
      super(scope, id, props);
    
    const vpc = props.vpc;

//    // Allow SSH (TCP Port 22) access from anywhere
//    const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', {
//     vpc,
//     description: 'Allow SSH (TCP port 22) in',
//     allowAllOutbound: true
//   });
  const securityGroup = props.securityGroup;
  securityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(22), 'Allow SSH Access')
  // securityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(8081), 'Allow HTTP 8081 port Access')
  // securityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), 'Allow HTTP Access')
  securityGroup.addIngressRule(securityGroup, ec2.Port.allTraffic(), 'Allow Self Access')

  const role = new iam.Role(this, 'ec2Role', {
    assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com')
  })

  role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'))
  role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3ReadOnlyAccess'))
  const customPolicyStatement = new iam.PolicyStatement({
    effect: iam.Effect.ALLOW,
    actions: ['lambda:GetFunctionConfiguration'], 
    resources: ['*'] 
  });
  role.addToPolicy(customPolicyStatement);

  const ami = new ec2.AmazonLinuxImage({
    generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2,
    cpuType: ec2.AmazonLinuxCpuType.X86_64
  });

  // Create the instance using the Security Group, AMI, and KeyPair defined in the VPC created
  const ec2Instance = new ec2.Instance(this, 'ProxyInstance', {
    vpc,
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
    machineImage: ami,
    securityGroup: securityGroup,
    vpcSubnets: {subnetType: ec2.SubnetType.PUBLIC,},
    role: role
  });

  // Create an asset that will be used as part of User Data to run on first load
  const asset = new Asset(this, 'UserdataAsset', { path: path.join(__dirname, '../ec2config.sh') });
  const localPath = ec2Instance.userData.addS3DownloadCommand({
    bucket: asset.bucket,
    bucketKey: asset.s3ObjectKey,
  });

  ec2Instance.userData.addExecuteFileCommand({
    filePath: localPath,
    arguments: '--verbose -y'
  });
  asset.grantRead(ec2Instance.role);

  this.instanceId = ec2Instance.instanceId;
  this.dnsName = ec2Instance.instancePublicDnsName;
  this.publicIP = ec2Instance.instancePublicIp;
    


    }
}