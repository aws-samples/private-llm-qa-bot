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


      const job = new glue.Job(this, 'chatbot-from-s3-to-aos',{
            executable: glue.JobExecutable.pythonShell({
            glueVersion: glue.GlueVersion.V1_0,
            pythonVersion: glue.PythonVersion.THREE_NINE,
            script: glue.Code.fromAsset(path.join(__dirname, '../../code/aos_write_job.py')),
          }),
          // jobName:'chatbot-from-s3-to-aos',
          maxConcurrentRuns:20,
          connections:[connection],
          maxCapacity:1,
          defaultArguments:{
              '--AOS_ENDPOINT':props.opensearch_endpoint,
              '--REGION':props.region,
              '--EMB_MODEL_ENDPOINT':process.env.embedding_endpoint,
              '--DOC_INDEX_TABLE':'chatbot_doc_index',
              '--additional-python-modules': 'pdfminer.six==20221105,gremlinpython==3.6.3,langchain==0.0.162,beautifulsoup4==4.12.2'
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
                ],
              effect: iam.Effect.ALLOW,
              resources: ['*'],
              })
      )
      this.jobArn = job.jobArn;
      this.jobName = job.jobName;
    
    }

}