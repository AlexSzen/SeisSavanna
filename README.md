# kenya


## How to work with Docker

### Get the image
#### Option 1: build the image

Be in the root directory of this repository. Running the following builds a Docker image tagged with the name `kenya`.

```
docker build -t kenya .
```

#### Option2 : pull the image

Not available yet.

### Run the image

This instantiates a container from the `kenya` Docker image and gets us an interactive terminal inside the container.
```
docker run --rm -it kenya
```

We can do the same and also mount a host file system folder to the container. The following mounts the current location where we're running the docker command as the `/data` directory within the container. Any changes in this directory will be persistent and reflected in the host file system.

```
docker run --rm -it -v $PWD:/data kenya
```
