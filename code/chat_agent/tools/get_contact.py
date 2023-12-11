import boto3
import json
from typing import Union

lambda_client= boto3.client('lambda')

def get_contact(**args) -> Union[str,None]:
    # print(args)
    payload = { "param" : args }
    invoke_response = lambda_client.invoke(FunctionName="employee_query_tool",
                                           InvocationType='RequestResponse',
                                           Payload=json.dumps(payload))

    response_body = invoke_response['Payload']
    response_str = response_body.read().decode("unicode_escape")
    response_str = response_str.strip('"')

    return response_str


if __name__ == "__main__":
    args = {"employee" : "Yun Li"}
    print(get_contact(**args))