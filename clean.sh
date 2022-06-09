#!/bin/bash

ps -ef | grep 'ray' | awk '{print $2}' | xargs kill -9
ps -ef | grep 'fps.x86' | awk '{print $2}' | xargs kill -9
ps -ef | grep 'python' | awk '{print $2}' | xargs kill -9
