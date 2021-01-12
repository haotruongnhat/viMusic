import redis
import time
import uuid
import json
import multiprocessing
import signal
import mpipe
import time

class ViQueue(object):
    def __init__(self,pipeline,global_lock,pool):
        #Connect to redis
        self._recv_db = redis.StrictRedis(connection_pool=pool)
        self._send_db = redis.StrictRedis(connection_pool=pool)
        self._pipeline = pipeline
        #handle signals for shutting down gracefully
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self._signal_exit = False
        self._global_lock = global_lock
    
    def start(self):
        #creating process
        #No lock needed
        self._send_proc = multiprocessing.Process(target=self._send)
        self._send_proc.start()
        self._recv_proc = multiprocessing.Process(target=self._recv)
        self._recv_proc.start()
        self.join()

    def join(self):
        self._send_proc.join()
        self._recv_proc.join()
        print("ViQueue closed gracefully!")

    def exit_gracefully(self,signum, frame):
        self._global_lock.acquire()
        self._signal_exit = True
        self._pipeline.put(None)
        self._global_lock.release()

    def _recv(self):
        while True:
            time.sleep(3)
            self._global_lock.acquire()
            queue = self._recv_db.lrange('music_dev', 0, 0) #FIXME: Problem with this one: no timeout
            self._recv_db.ltrim('music_dev', 1, -1)
            self._global_lock.release()
            for q_redis in queue:
                q_redis = json.loads(q_redis.decode("utf-8"))
                print(q_redis)
                self._pipeline.put(q_redis)
                print("Received")
            if self._signal_exit:
                print("Shutting down send process")
                self._send_proc.terminate()
                break

    def _send(self):
        for q_redis in self._pipeline.results():
            if q_redis['status'] == 'fail':
                del q_redis["danceability"]
                del q_redis["energy"]
                del q_redis["speechiness"]
                del q_redis["valence"]
                del q_redis["acousticness"]
                del q_redis["instrumentalness"]
                del q_redis["liveness"]
                self._global_lock.acquire()
                self._send_db.set(q_redis['idProcess'],json.dumps(q_redis))
                self._global_lock.release()
                continue
            #Delete unnecessary keys
            del q_redis["time_signature_numerator"]
            del q_redis["time_signature_denominator"]
            del q_redis["key_signature"]
            del q_redis["bpm"]
            del q_redis["danceability"]
            del q_redis["energy"]
            del q_redis["speechiness"]
            del q_redis["valence"]
            del q_redis["acousticness"]
            del q_redis["instrumentalness"]
            del q_redis["liveness"]
            del q_redis["primer_path"]
            del q_redis['pop_stresses']
            del q_redis['pop_syllables']
            del q_redis['pop_section_groups']
            del q_redis['pop_boundaries']
            del q_redis['rock_stresses']
            del q_redis['rock_syllables']
            del q_redis['rock_section_groups']
            del q_redis['rock_boundaries']
            del q_redis['electro_stresses']
            del q_redis['electro_syllables']
            del q_redis['electro_section_groups']
            del q_redis['electro_boundaries']

            self._global_lock.acquire()
            q_redis['status'] = 'done'
            print(q_redis)
            self._send_db.set(q_redis["idProcess"], json.dumps(q_redis))
            print("{}: Sent".format(q_redis["idProcess"]))
            self._global_lock.release()

            if self._signal_exit:
                print("Shutting down receiving")
                self._recv_proc.terminate()
                break


    def results(self):
        return self._pipeline.results()
