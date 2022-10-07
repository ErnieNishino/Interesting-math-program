m = int(input("min:")) #min
n = int(input("max:")) #max
l = []


#judge whether num is prime
def is_prime(num):
    for n in range(2,int(num**0.5)+1):
        if num%n==0:
            return False
    return True


for k in range(m,n):
    if is_prime(k):
        l.append(k)
        
print(l)

