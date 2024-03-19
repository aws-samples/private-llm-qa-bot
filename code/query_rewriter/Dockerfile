FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.9

COPY ./requirements.txt  .

RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY query_rewrite.py ${LAMBDA_TASK_ROOT}
RUN mkdir -p ${LAMBDA_TASK_ROOT}/generator
COPY generator/*.py ${LAMBDA_TASK_ROOT}/generator/
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "query_rewrite.lambda_handler" ]