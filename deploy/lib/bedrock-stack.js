import * as path from 'path';
import {
  App,
  Stack,
  aws_iam as iam,
  aws_opensearchserverless as openSearch,
  aws_lambda as lambda,
  Duration,
  CustomResource,
  aws_logs as logs,
  custom_resources,
  CfnOutput,
} from 'aws-cdk-lib';
import { BedrockKnowledgeBase } from 'bedrock-agents-cdk';
import { fileURLToPath } from "url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class BedrockCdkStack extends Stack {
  /**
   *
   * @param {Construct} scope
   * @param {string} id
   * @param {StackProps=} props
   */
  constructor(scope, id, props) {
    super(scope, id, props);

    const region = props.env.region;
    const s3bucket = props.env.s3bucket;
    const account_id = Stack.of(this).account;

        // Change the resources' name or use parameters
    // https://docs.aws.amazon.com/cdk/v2/guide/parameters.html
    const collectionName = 'chatbot-kd';
    const indexName = 'chatbot-index';
    // user who can access the OpenSearch Dashboard
    const adminUserArn = `arn:aws:iam::${account_id}:role/admin_for_chatbot`;

    const kbRoleArn = new iam.Role(this, 'BedrockKnowledgeBaseRole', {
      roleName: 'AmazonBedrockExecutionRoleForKnowledgeBase_kb_test',
      assumedBy: new iam.ServicePrincipal('bedrock.amazonaws.com'),
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess')],
    }).roleArn;

    const customResourceRole = new iam.Role(this, 'CustomResourceRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole')],
    });

    // OpenSearch collection
    const collection = new openSearch.CfnCollection(this, 'OpenSearchCollection', {
      name: `${collectionName}`,
      type: 'VECTORSEARCH',
      description: `${collectionName}`
    });

    const encryptionPolicy = new openSearch.CfnSecurityPolicy(this, 'EncryptionPolicy', {
      name: 'embeddings-encryption-policy',
      type: 'encryption',
      description: `Encryption policy for ${collectionName} collection.`,
      policy: `
      {
        "Rules": [
          {
            "ResourceType": "collection",
            "Resource": ["collection/${collectionName}*"]
          }
        ],
        "AWSOwnedKey": true
      }
      `,
    });

    // Opensearch network policy
    const networkPolicy = new openSearch.CfnSecurityPolicy(this, 'NetworkPolicy', {
      name: 'embeddings-network-policy',
      type: 'network',
      description: `Network policy for ${collectionName} collection.`,
      policy: `
        [
          {
            "Rules": [
              {
                "ResourceType": "collection",
                "Resource": ["collection/${collectionName}*"]
              },
              {
                "ResourceType": "dashboard",
                "Resource": ["collection/${collectionName}*"]
              }
            ],
            "AllowFromPublic": true
          }
        ]
      `,
    });

    // Opensearch data access policy
    const dataAccessPolicy = new openSearch.CfnAccessPolicy(this, 'DataAccessPolicy', {
      name: 'embeddings-access-policy',
      type: 'data',
      description: `Data access policy for ${collectionName} collection.`,
      policy: `
        [
          {
            "Rules": [
              {
                "ResourceType": "collection",
                "Resource": ["collection/${collectionName}*"],
                "Permission": [
                  "aoss:CreateCollectionItems",
                  "aoss:DescribeCollectionItems",
                  "aoss:DeleteCollectionItems",
                  "aoss:UpdateCollectionItems"
                ]
              },
              {
                "ResourceType": "index",
                "Resource": ["index/${collectionName}*/*"],
                "Permission": [
                  "aoss:CreateIndex",
                  "aoss:DeleteIndex",
                  "aoss:UpdateIndex",
                  "aoss:DescribeIndex",
                  "aoss:ReadDocument",
                  "aoss:WriteDocument"
                ]
              }
            ],
            "Principal": [
              "${customResourceRole.roleArn}",
              "${kbRoleArn}"
            ]
          }
        ]
      `,
    });

    const collectionArn = collection.attrArn
    const collection_endpoint = collection.attrCollectionEndpoint
    new CfnOutput(this,'collectionArn', {value:collectionArn});
    new CfnOutput(this,'collectionEndpoint',{value:collection_endpoint});

    const lambdaLayerZipFilePath = '../lambda_layer/bedrock-agent-layer.zip';
    const customResourcePythonFilePath = '../custom_resource';
    const textField = 'content'
    const vectorFieldName = 'embedding'
    const metadataField = 'metadata-field'

    const layer = new lambda.LayerVersion(this, 'OpenSearchCustomResourceLayer', {
      code: lambda.Code.fromAsset(path.join(__dirname, lambdaLayerZipFilePath)),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_10],
      description: 'Required dependencies for Lambda',
    });

    customResourceRole.addToPolicy(
      new iam.PolicyStatement({
        resources: [ collectionArn ],
        actions: ['aoss:APIAccessAll'],
      }),
    );

    // Lambda function
    const onEventFunction = new lambda.Function(this, 'OpenSearchCustomResourceFunction', {
      runtime: lambda.Runtime.PYTHON_3_10,
      handler: 'indices_custom_resource.on_event',
      code: lambda.Code.fromAsset(path.join(__dirname, customResourcePythonFilePath)),
      layers: [layer],
      timeout: Duration.seconds(600),
      environment: {
        COLLECTION_ENDPOINT: collection_endpoint,
        VECTOR_INDEX_NAME: indexName,
        VECTOR_FIELD_NAME: vectorFieldName,
        TEXT_FIELD: textField,
        METADATA_FIELD: metadataField,
      },
      role: customResourceRole,
    });

    // Custom resource provider
    const provider = new custom_resources.Provider(this, 'CustomResourceProvider', {
      onEventHandler: onEventFunction,
      logRetention: logs.RetentionDays.ONE_DAY,
    });

    // Custom resource
    const cr_obj = new CustomResource(this, 'CustomResource', {
      serviceToken: provider.serviceToken,
    });

    const dataSourceBucketArn = `arn:aws:s3:::${s3bucket}`

    new CfnOutput(this,'BedrockKnowledgeBaseSourceArn',{value:dataSourceBucketArn});

    const knowledgeBase = new BedrockKnowledgeBase(this, 'BedrockOpenSearchKnowledgeBase', {
      name: 'chatbot-bedrock-kd',
      roleArn: kbRoleArn,
      storageConfiguration: {
        opensearchServerlessConfiguration: {
          collectionArn: collectionArn,
          fieldMapping: {
            metadataField: metadataField,
            textField: textField,
            vectorField: vectorFieldName,
          },
          vectorIndexName: indexName,
        },
        type: 'OPENSEARCH_SERVERLESS',
      },
      dataSource: {
        name: 'chatbot-datasrc',
        dataSourceConfiguration: {
          s3Configuration: {
            bucketArn: dataSourceBucketArn,
            inclusionPrefixes: [ 'bedrock-kb-src' ]
          },
          type: 'S3',
        },
      },
    });

    const knowledgeBaseId = knowledgeBase.knowledgeBaseId

    new CfnOutput(this,'BedrockKnowledgeBaseArn',{value:knowledgeBase.roleArn});
    new CfnOutput(this,'knowledgeBaseId',{value:knowledgeBaseId});

    collection.addDependency(encryptionPolicy);
    collection.addDependency(networkPolicy);
    collection.addDependency(dataAccessPolicy);
    cr_obj.node.addDependency(collection);
    knowledgeBase.node.addDependency(cr_obj);

  }
}

