FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.9

COPY ./requirements.txt  .

RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY lambda.py ${LAMBDA_TASK_ROOT}
COPY llm_wrapper.py ${LAMBDA_TASK_ROOT}
COPY llm_manager.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda.lambda_handler" ]