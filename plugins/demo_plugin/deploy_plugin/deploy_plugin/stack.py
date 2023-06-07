from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_ec2 as ec2,
    CfnOutput
)
from constructs import Construct
import aws_cdk as cdk
import os
from dotenv import load_dotenv

class DeployPluginStack(Stack):

  def __init__(self, scope: Construct, id: str, vpc_id: str,caller_role_name:str, table_name: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        vpc = ec2.Vpc.from_lookup(self, "MyVPC", vpc_id=vpc_id)

        # 创建存放todolist的table 
        dynamodb_table = dynamodb.Table(
            self,
            "MyDynamoDBTable",
            table_name=table_name,
            partition_key=dynamodb.Attribute(name="username", type=dynamodb.AttributeType.STRING)
        )

        # 创建Lambda function
        lambda_function = lambda_.Function(
            self,
            "pluginfunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="app.lambda_handler",
            code=lambda_.Code.from_asset("../code"),
            vpc=vpc,
            environment={
                "TABLE_NAME": dynamodb_table.table_name
            }
        )

        # grant Lambda function permissions to access DynamoDB table
        dynamodb_table.grant_read_write_data(lambda_function)


        caller_lambda_role = iam.Role.from_role_name(
            self,
            "CallerRole",
            role_name= caller_role_name,
        )

        fn_url = lambda_function.add_function_url()
        fn_url.grant_invoke_url(caller_lambda_role)

        CfnOutput(self, "The lambda function Url",
            value=fn_url.url
        )

        # # create an IAM role that allows the Lambda function to be invoked via AWS_IAM authorization
        # lambda_role = iam.Role(
        #     self,
        #     "MyLambdaRole",
        #     assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
        # )
        # lambda_role.add_to_policy(
        #     iam.PolicyStatement(
        #         actions=["lambda:InvokeFunction"],
        #         resources=[lambda_function.function_arn],
        #         effect=iam.Effect.ALLOW
        #     )
        # )

        # # allow the Lambda function to assume the IAM role
        # lambda_function.add_to_role_policy(
        #     iam.PolicyStatement(
        #         actions=["sts:AssumeRole"],
        #         resources=[lambda_role.role_arn],
        #         effect=iam.Effect.ALLOW
        #     )
        # )
