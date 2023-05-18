# sample dockerfile for lambda container with Python 3.8
FROM public.ecr.aws/lambda/python:3.8

# Copy function code
COPY os_search.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt  .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "os_search.handler" ]