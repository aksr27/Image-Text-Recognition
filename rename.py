import os
path = os.getcwd()+"/Bmp"
# print(path)
# print(os.listdir(path))
i=0
x=os.listdir(path)
x.sort()
print(x)
for filename in x:
	path2 = os.getcwd()+"/Bmp/"+str(filename)
	y=os.listdir(path2)
	j=3000
	print(y)
	for imgname in y:
		print(imgname)
		os.rename(os.path.join(path2,imgname), os.path.join(path2,chr(i+65)+"_"+str(j)+".png"))
		j+=1
	os.rename(os.path.join(path,filename), os.path.join(path,chr(i+65)))
	i = i +1
	if(i==26):
		break
print(os.listdir(path))


# print(chr(97))