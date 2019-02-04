
help:
	cat Makefile
run:
	docker-compose up
exec:
	docker-compose exec bot_cpbr bash
build: stop .FORCE
	docker-compose build
rebuild: stop .FORCE
	docker-compose build --force-rm
stop:
	docker stop bot_cpbr || true; docker rm bot_cpbr || true;
.FORCE:
