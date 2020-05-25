# -*- coding: UTF-8 -*-

import time
import pickle
import os

def LoadPickleFromFile(localFile):
		try:
				f = open(localFile,'rb')
		except OSError:
				print('本地pickle文件：%s 读取失败' %(localFile))
				return -1
		myValue = pickle.load(f)
		f.close()
		print('本地pickle文件：%s 读取成功' %(localFile))
		return myValue

def SavePickleToFile(myValue,localFile):
		try:
				f = open(localFile,'wb') #覆盖原内容
		except OSError:
				print('本地pickle文件：%s 写入失败' %(localFile))
				return -1

		pickle.dump(myValue, f)
		f.close()
		print('本地pickle文件：%s 写入成功' %(localFile))
		return 1



if __name__ == "__main__":
	localPickleFile = 'pickleFile'
	lastvalue = LoadPickleFromFile(localPickleFile)
	if(lastvalue != -1):
		print("Last Info. = %s" % lastvalue)
	inputStr = input('input Info. needed to save: ')
	SavePickleToFile(inputStr,localPickleFile)