import {NestedStack,CfnOutput} from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ops from 'aws-cdk-lib/aws-opensearchserverless';

export class OpenSearchServerlessStack extends NestedStack {
  domainEndpoint;

  constructor(scope, id, props) {
    super(scope, id, props);
    const searchConstruct = new OpenSearchConstruct(this, 'OpenSearchConstruct',props);
    this.domainEndpoint = searchConstruct.domainEndpoint
  }
}

class OpenSearchConstruct extends Construct {

  domainEndpoint;
  

  constructor(scope, id,props) {
    super(scope, id);

    // See https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-manage.html
    const collection = new ops.CfnCollection(this, 'ProductSearchCollection', {
      name: 'kb-collection',
      type: 'VECTORSEARCH',
    });

    // Encryption policy is needed in order for the collection to be created
    const encPolicy = new ops.CfnSecurityPolicy(this, 'ProductSecurityPolicy', {
      name: 'kb-collection-policy',
      policy: '{"Rules":[{"ResourceType":"collection","Resource":["collection/kb-collection"]}],"AWSOwnedKey":true}',
      type: 'encryption'
    });
    collection.addDependency(encPolicy);

    // Network policy is required so that the dashboard can be viewed!
    const netPolicy = new ops.CfnSecurityPolicy(this, 'ProductNetworkPolicy', {
      name: 'kb-network-policy',
      policy: `[{"Rules":[{"ResourceType":"collection","Resource":["collection/kb-collection"]}, {"ResourceType":"dashboard",
      "Resource":["collection/kb-collection"]}],"AllowFromPublic":false,"SourceVPCEs":["${props.vpc.vpcId}"]}]`,
      type: 'network'
    });
    collection.addDependency(netPolicy);

    this.domainEndpoint = collection.attrCollectionEndpoint;

    new CfnOutput(this, 'DashboardEndpoint', {
      value: collection.attrDashboardEndpoint,
    });
    new CfnOutput(this, 'AosServerlessEndpoint', {
      value: collection.attrCollectionEndpoint,
    });
  }
}