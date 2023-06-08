#!/usr/bin/env python3
import os

import aws_cdk as cdk
from dotenv import load_dotenv

from deploy_plugin.stack import DeployPluginStack

assert load_dotenv()

print(os.getenv('CDK_DEFAULT_ACCOUNT'),os.getenv('CDK_DEFAULT_REGION'))
app = cdk.App()
DeployPluginStack(app, 
                "MyLambdaStack",
                vpc_id="vpc-08804dc0c72657e28",    
                table_name="todolist",
                caller_role_name = 'todos_plugin-role-7rvbhxbc',
                env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), 
                region=os.getenv('CDK_DEFAULT_REGION')),
            )
app.synth()
