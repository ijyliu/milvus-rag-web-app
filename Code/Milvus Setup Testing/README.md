# set up milvus tutorial

- im going to go through how to set all this up and how to use the files in here

## step 1: install milvus

1. go to this [website](https://milvus.io/docs/v2.0.x/install_standalone-docker.md) and follow the instrcutions

- NOTE: you will notice that this folder already has a docker-compose.yml. you can probably skip the part about downloading the yml on the website, but you will want to run `docker compose up -d` from your terminal wherever the .yml file is located

2. once you have run this, open docker desktop and you should now see an image in there called milvus. this is the docker image, and when you want to use milvus you will have to make sure the image is running in docker desktop. look on the right where it says Actions and you should see a play button. when you want to use milvus, press the play button. this is what will start the image, and if the image is not running then your connections to milvus will not work

- NOTE: there are ways to run this from the command line and automate the turning on and off of the image but its more work and this is the easiest way to do this.

3. install pymilvus w/ command line, `pip install pymilvus`

- NOTE: when i was doing this i ran into version issues for lawd knows what reason and I have to run
  `pip uninstall pymilvus`
  `pip install pymilvus`
  in my command line to fix it. if you run into problems ask chatgpt it is pretty good at that.

## step 2: explore the sample code in these folders

- i have set up some super basic code here, they work as follows.
- main_test.py: tests connecting to milvus. thats it
- gen_test_embed.py: generates a bunch of embeddings. these are already saved in dummy_data.json. fi you want to generate your own embeddings, feel free to use this as a template
- load_test_collect.py: loads the vectors from dummy_data.json into the milvus data base.
  - NOTE: run this one or main.py first, it will load all the vectors. if you run read_collect.py, note that your vectors will most likely have different ids, so change the id you are querying on (if your vectors have the same ideas, let me know because i would find that kinda interesting)
- read_test_collect.py: reads the vectors out of the milvus database
  - NOTE: if you want to actually run this code, you will have to run `pip install spacy` in your command line and you will have to have your milvus container running in docker desktop
