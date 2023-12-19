import { CfnOutput, NestedStack,RemovalPolicy }  from 'aws-cdk-lib';
import {EngineVersion,Domain} from 'aws-cdk-lib/aws-opensearchservice';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from "aws-cdk-lib/aws-iam";


export class OpenSearchStack extends NestedStack {
    domainEndpoint;
    domain;
    sg;

/**
   *
   * @param {Construct} scope
   * @param {string} id
   * @param {StackProps=} props
   */
constructor(scope, id, props) {
    super(scope, id, props);


    const devDomain = new Domain(this, 'Domain', {
        version: EngineVersion.openSearch('2.11'),
        removalPolicy: RemovalPolicy.DESTROY,
        vpc:props.vpc,
        zoneAwareness: {
          enabled:true
        },
        securityGroups: [props.securityGroup],
        capacity: {
            dataNodes: 2,
            dataNodeInstanceType:'r6g.large.search'
          },
        ebs: {
        volumeSize: 300,
        volumeType: ec2.EbsDeviceVolumeType.GENERAL_PURPOSE_SSD_GP3,
        },
      });

      devDomain.addAccessPolicies(new iam.PolicyStatement({

        actions: ['es:*'],
        effect: iam.Effect.ALLOW,
        principals:[new iam.AnyPrincipal()],
        resources: [`${devDomain.domainArn}/*`],
      }))


      this.domainEndpoint = devDomain.domainEndpoint;
      this.domain = devDomain;

    

}
}