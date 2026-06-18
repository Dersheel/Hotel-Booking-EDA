print('Welcome to my computer quiz!')
playing=input("Do you want to play? ").lower()

if playing != 'yes':
    quit()
print("Let's play")

score=0
answer=input("What is the capital of India? ").lower()
if answer == "delhi":
    print("Correct Answer")
    score += 1
else:
    print("Incorrect Answer")

answer=input("What does CPU stand for? ").lower()
if answer == "central processing unit":
    print("Correct Answer")
    score += 1
else:
    print("Incorrect Answer")

answer=input("What does RAM stand for? ").lower()
if answer == "random access memory":
    print("Correct Answer")
    score += 1
else:
    print("Incorrect Answer")

answer=input("What is the full form of SQL? ").lower()
if answer == "structured query language":
    print("Correct Answer")
    score += 1
else:
    print("Incorrect Answer")

print("you have answered " + str(score) + " questions correctly!")
print("you have scored " + str((score/4)*100) + "%.")
