import { Stack, Duration, RemovalPolicy, CfnOutput } from "aws-cdk-lib";
import { AttributeType, Table } from "aws-cdk-lib/aws-dynamodb";
import { LambdaStack } from "./frontend-lambda-stack.js";
import * as dotenv from "dotenv";

dotenv.config();

export class FrontendCdkStack extends Stack {
  /**
   *
   * @param {Construct} scope
   * @param {string} id
   * @param {StackProps=} props
   */
  constructor(scope, id, props) {
    super(scope, id, props);

    const user_table = new Table(this, "user_table", {
      partitionKey: {
        name: "username",
        type: AttributeType.STRING,
      },
      tableName :"chatbotFE_user",
      removalPolicy: RemovalPolicy.DESTROY, // NOT recommended for production code
    });
   
    const lambdastack = new LambdaStack(this, "lambdas", {
      user_table,
    });

    new CfnOutput(this, `API gateway endpoint url`, {
      value: `${lambdastack.apigw_url}`,
    });

    new CfnOutput(this, "ChatBotWsApi_URL", {
      value: lambdastack.webSocketURL,
    });

  }
}
