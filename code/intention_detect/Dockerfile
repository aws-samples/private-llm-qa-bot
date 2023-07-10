FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.9

COPY ../requirements.txt  .
COPY ../boto3-1.26.140-py3-none-any.whl .
COPY ../botocore-1.29.140-py3-none-any.whl .

RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
RUN  pip3 install botocore-1.29.140-py3-none-any.whl --target "${LAMBDA_TASK_ROOT}"
RUN  pip3 install boto3-1.26.140-py3-none-any.whl --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY intention.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "intention.lambda_handler" ]
