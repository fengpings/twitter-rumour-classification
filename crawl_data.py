import tweepy
import json
import os
import pickle
import time

consumer_key = "jKGBLZmNGf2JZ4e8BR7jEcURk"

consumer_secret = "2rN32C4gJO9FZ45bDwcSuwrD2qxXOgJ9N7fryu4MJPYCGoEA34"

access_key = "1514428987145351169-DqaLst7SRyqyCN3WjSEI0glb4U65SU"

access_secret = "jra6WsoJPmANoCPaW7eJg6gWuHZw960RN4WvS6KAEe3hV"

# bearer_token = "AAAAAAAAAAAAAAAAAAAAAInkbQEAAAAA4qpV4fVfNO4mQVgAuEdKGJ6mvFU%3Darb2mAKTUVh9pmGltG5du7ilhgYvhotz4xux0WQPych06cqIbd"

# client = tweepy.Client(bearer_token=bearer_token)
# client = tweepy.Client(
#     consumer_key=consumer_key, consumer_secret=consumer_secret,
#     access_token=access_token, access_token_secret=access_token_secret
# )
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)
invalid = 0

c_ids = os.listdir('./covid')
cralwed = [x.split('.')[0].strip() for x in c_ids]

id = []
with open('covid.data.txt') as f:
    ids = f.readlines()
    for i in ids:
        id.extend([i.strip() for i in i.split(',')])
if 'no_status.pkl' in os.listdir('./'):
    with open('no_status.pkl','rb') as file:
        no_status = pickle.load(file)
else:
    no_status = []
rest = [i for i in id if i not in cralwed and i not in no_status]
print(len(rest))
for i in rest:
    try:
        dict_json = json.dumps(api.get_status(id=int(i.strip()))._json)
        with open(f'covid/{i.strip()}.json', 'w+') as file:
            file.write(dict_json)
    except Exception as e:
        invalid += 1
        print(f'========{invalid}=========')
        print(e)
        if '429' in str(e):
            time.sleep(900)
            try:
                dict_json = json.dumps(api.get_status(id=int(i.strip()))._json)
                with open(f'covid/{i.strip()}.json', 'w+') as file:
                    file.write(dict_json)
            except Exception as e:
                invalid += 1
                print(f'========{invalid}=========')
                print(e)
                if 'Failed to send request' in str(e).strip():
                    with open('no_status.pkl', 'wb') as file:
                        pickle.dump(no_status, file)
                    break
                no_status.append(no_status)
                continue
        elif 'Failed to send request' in str(e).strip():
            with open('no_status.pkl', 'wb') as file:
                pickle.dump(no_status, file)
            break
        else:
            no_status.append(i)
        continue

with open('no_status.pkl', 'wb') as file:
    pickle.dump(no_status, file)
# a = api.get_status(id=544289941996326912)
