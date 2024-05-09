import * as glue from  '@aws-cdk/aws-glue-alpha';
import { NestedStack,Duration, CfnOutput }  from 'aws-cdk-lib';
import * as iam from "aws-cdk-lib/aws-iam";
import * as dotenv from "dotenv";
dotenv.config();
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class GlueStack extends NestedStack {

    jobArn = '';
    jobName = '';
    /**
     *
     * @param {Construct} scope
     * @param {string} id
     * @param {StackProps=} props
     */
    constructor(scope, id, props) {
      super(scope, id, props);


      const connection = new glue.Connection(this, 'GlueJobConnection', {
        type: glue.ConnectionType.NETWORK,
        vpc: props.vpc,
        securityGroups: props.securityGroups,
        subnet:props.subnets[0],
      });


      const job = new glue.Job(this, 'chatbot-ingestion-job',{
            executable: glue.JobExecutable.pythonShell({
            glueVersion: glue.GlueVersion.V1_0,
            pythonVersion: glue.PythonVersion.THREE_NINE,
            script: glue.Code.fromAsset(path.join(__dirname, '../../code/offline_process/aos_write_job.py')),
          }),
          jobName:'chatbot-ingestion-job',
          maxConcurrentRuns:100,
          maxRetries:3,
          connections:[connection],
          maxCapacity:1,
          defaultArguments:{
              '--AOS_ENDPOINT':props.opensearch_endpoint,
              '--REGION':props.region,
              '--EMB_MODEL_ENDPOINT':process.env.embedding_endpoint,
              '--DOC_INDEX_TABLE':'chatbot_doc_index',
              '--additional-python-modules': 'pdfminer.six==20221105,gremlinpython==3.6.3,langchain==0.0.162,beautifulsoup4==4.12.2,boto3>=1.28.52,botocore>=1.31.52,,anthropic_bedrock,python-docx',
              '--company' : 'default',
              '--emb_batch_size' : '20'
          }
      })
      job.role.addToPrincipalPolicy(
        new iam.PolicyStatement({
              actions: [ 
                "sagemaker:InvokeEndpointAsync",
                "sagemaker:InvokeEndpoint",
                "s3:List*",
                "s3:Put*",
                "s3:Get*",
                "es:*",
                "dynamodb:*",
                "bedrock:*",
                ],
              effect: iam.Effect.ALLOW,
              resources: ['*'],
              })
      )

      const update_intention_index_job = new glue.Job(this, 'update_intention_index_job',{
            executable: glue.JobExecutable.pythonShell({
            glueVersion: glue.GlueVersion.V1_0,
            pythonVersion: glue.PythonVersion.THREE_NINE,
            script: glue.Code.fromAsset(path.join(__dirname, '../../code/offline_process/update_intention_index_regular.py')),
          }),
          jobName:'update_intention_index_job',
          maxConcurrentRuns:1,
          maxRetries:0,
          connections:[connection],
          maxCapacity:1,
          defaultArguments:{
              '--additional-python-modules': 'boto3>=1.28.52,botocore>=1.31.52,pytz,requests_aws4auth,opensearch-py',
              '--bucket':process.env.UPLOAD_BUCKET,
              '--region':props.region,
              '--aos_endpoint':props.opensearch_endpoint,
              '--emb_model_endpoint':process.env.embedding_endpoint,
              '--path_prefix':'intention/',
              '--ssm_key_for_index_status':'intention_index_status',
              '--concurrent_runs_quota':'10',
              '--job_name':'chatbot-ingestion-job',
              '--company':"default",
              '--emb_batch_size':'20'
          }
      })
      update_intention_index_job.role.addToPrincipalPolicy(
        new iam.PolicyStatement({
              actions: [ 
                "s3:List*",
                "s3:Put*",
                "s3:Get*",
                "es:*",
                "dynamodb:*",
                "ssm:*"
                ],
              effect: iam.Effect.ALLOW,
              resources: ['*'],
              })
      )

      this.jobArn = job.jobArn;
      this.jobName = job.jobName;
    
    }

}