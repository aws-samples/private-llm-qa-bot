import aws_cdk as core
import aws_cdk.assertions as assertions

from deploy_plugin.deploy_plugin_stack import DeployPluginStack

# example tests. To run these tests, uncomment this file along with the example
# resource in deploy_plugin/deploy_plugin_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = DeployPluginStack(app, "deploy-plugin")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
