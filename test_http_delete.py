import re

line = "Pre-movie line coffee run. (@ Starbucks) http://4sq.com/5EA7w3"
print line

matchObj = re.match(r'(.*)http(.*?)//*', line, re.M | re.I)

if matchObj:
    print "matchObj.group() : ", matchObj.group()
    print "matchObj.group(1) : ", matchObj.group(1)
    print "matchObj.group(2) : ", matchObj.group(2)
else:
    print "No match!!"