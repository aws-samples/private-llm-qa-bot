import {NestedStack,CfnOutput,Stack} from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ops from 'aws-cdk-lib/aws-opensearchserverless';
import * as dotenv from "dotenv";
dotenv.config();

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
    const account_id = Stack.of(this).account;
    const role_name = process.env.role_name??'admin_role'

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
      policy: `[{"Rules":[{"ResourceType":"collection","Resource":["collection/kb-collection"]}, {"ResourceType":"dashboard","Resource":["collection/kb-collection"]}],"AllowFromPublic":true}]`,
      type: 'network'
    });
    collection.addDependency(netPolicy);

    //Data policy
    const cfnAccessPolicy = new ops.CfnAccessPolicy(this, 'AossDataAccessPolicy', {
      name: 'kb-data-policy',
      policy: `[{"Rules":[{"ResourceType":"index","Resource":["index/*/*"],"Permission":["aoss:*"]},{"ResourceType":"collection","Resource":["collection/kb-collection"],"Permission":["aoss:*"]}],"Principal":["arn:aws:iam::${account_id}:role/admin_role"]}]`,
      type: 'data',
    });

    this.domainEndpoint = collection.attrCollectionEndpoint;

    new CfnOutput(this, 'DashboardEndpoint', {
      value: collection.attrDashboardEndpoint,
    });
    new CfnOutput(this, 'AosServerlessEndpoint', {
      value: collection.attrCollectionEndpoint,
    });
  }
}