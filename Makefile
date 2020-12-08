
DEMO_STEM=edgerealtimevideoanalytics
SPEC=docker-compose.local.yaml
SUDO=sudo

ifeq ($(GPU),1)
GPU_DOCKER=docker
GPU_OPT=--gpus all
else
GPU_DOCKER=docker
GPU_OPT=
endif

start-edge:
	@$(DOCKER) run --rm --name edge --network host -it $(GPU_OPT) $(DEMO_STEM)_redisedge:latest

start:
	@docker-compose -f $(SPEC) up

start-all:
	@$(DOCKER) run --rm -d --name edge --network host -it $(GPU_OPT) $(DEMO_STEM)_redisedge:latest
	@docker-compose -f $(SPEC) up -d --force-recreate

stop-all:
	@docker-compose -f $(SPEC) down
	@docker stop edge

logs:
	@docker logs edge

build:
	@docker-compose -f docker-compose.yaml build

setup:
	@git lfs install; git lfs fetch; git lfs checkout
	@$(SUDO) ./sbin/getcompose

.PHONY: start-edge start start-all stop stop-all logs build setup
