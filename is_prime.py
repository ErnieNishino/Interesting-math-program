import math
m = int(input("m:")) #min
n = int(input("n:")) #max
l = []


#judge whether num is prime
def is_p(num):
    a = 1
    n = math.sqrt(num)
    for i in range(2,int(n)+1):
        if num%i == 0:
            a = 0
            break
    return a


for k in range(m,n):
    if is_p(k):
        l.append(k)
        
print(l)

#write numbers into a txt
with open("pirme.txt","w") as f:
    cnt = 0
    for i in l:
        if cnt == 20:
            f.write("\n")
            cnt=0
        f.write(str(i))
        f.write(" ")
        cnt+=1
