
import { NestedStack, CfnOutput,RemovalPolicy }  from 'aws-cdk-lib';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import { aws_elasticloadbalancingv2_targets as elasticloadbalancingv2_targets } from 'aws-cdk-lib';
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam'
import * as path from 'path';

export class ALBStack extends NestedStack {

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

    
    const alb = new elbv2.ApplicationLoadBalancer(this, 'Alb', {
        vpc,
        internetFacing: true,
    });

     // Export the ALB DNS name so that it can be used by other stacks.
     this.dnsName = alb.loadBalancerDnsName;

    const listener = alb.addListener('HTTPListener',{
        port: 80,
        protocol: elbv2.Protocol.HTTP,
      });

    
    const instanceIdTarget = new elasticloadbalancingv2_targets.InstanceIdTarget(props.instanceId);


    const opensearchproxyTargetGroup = new elbv2.ApplicationTargetGroup(this, 'OpenSearchTargetGroup', {
        vpc,
        port: 443, 
        targetType:elbv2.TargetType.INSTANCE,
        targets:[instanceIdTarget]
      });

    listener.addTargetGroups('OpenSearchTargets', {
        targetGroups: [opensearchproxyTargetGroup],
        port: 443,
      });

      listener.connections.allowDefaultPortFromAnyIpv4('Open to the world');

    


    }
}