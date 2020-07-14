import fire
import os


def execute(command):
	command = command.replace('+', ' ')
	# os.system(command)
	print(command)


if __name__ == '__main__':
	fire.Fire(execute)
