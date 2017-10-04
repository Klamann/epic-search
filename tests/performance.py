import random
import time
from threading import Thread
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests

base_url = "http://localhost:1337/"
search_url = base_url + "search"

# terms from http://marvin.cs.uidaho.edu/Teaching/CS112/terms.pdf
query_terms = """
AI, ASCII, Abstraction, Address, Agent, Based, Modeling, Agent, Algorithm, Analog, Application, Argument, Artificial, Intelligence, Autonomous, Bandwidth, Big, Data, Binary, Bit, Boot, Program, Boot, Broadcast, Bus, Byte, Call, CPU, intensive, CPU, Cache, Card, Central, Processing, Unit, Character, Chip, Clock, Cloud, Computing, Cloud, Storage, Cloud, Cluster, Computer, Compiler, Computer, science, Computer, Control, structure, Core, Customer, Base, Data, Mining, Data, intensive, Database, Device, Digital, Divide, Digital, Disk, Drone, Emacs, Executable, External, hard, drive, File, Extension, File, Floating, Point, Number, Functional, Abstraction, Function, Global, variable, Graphics, card, Graphics, processing, HSV, HTML, HW, Hard, drive, Hardware, Hertz, Hexadecimal, Hex, High, Level, Language, I/O, IDE, IT, Instruction, Integer, Interface, Interpreter, Kilobyte, Linux, Local, variable, Loop, Machine, Instructions, Memory, Message, Passing, Motherboard, Multi-Core, processor, Northbridge, OS, Object, Oriented, programming, Object, Octal, On, Chip, Open, Source, Operating, System, PC, Pairs, Programming, Parallel, Computing, Parallel, Peripheral, Port, Powers, of, Two, Procedure, Process, Programming, Program, Proprietary, Protocol, RAM, RGB, Reboot, STEM, SW, Scalable, Computing, Semantics, Server, Simulation, Software, Solid, state, disk, (SSD), Southbridge, Speed, of, Light, String, Supercomputer, Syntax, Telepresence, Text, Thread, Time, sharing, Type, UAV, UNIX, USB, Unicode, Unicode, Virtual, Machine, Visibility, Vi, Windows
"""
query_terms = [t.strip() for t in query_terms.split(",")]


def random_query(min_terms=1, max_terms=3):
    num_terms = random.randint(min_terms, max_terms)
    return " ".join(random.choice(query_terms) for _ in range(num_terms))


def run_queries_seq(num_queries=10, same_session=False):
    response_times = []
    sid = ""
    for i in range(num_queries):
        q = random_query()
        r = requests.get(search_url, params={'q': q, 'sid': sid})
        delta = r.elapsed.total_seconds()
        response_times.append(delta)
        if same_session and not sid:
            sid = parse_qs(urlparse(r.url).query)['sid'][0]
        print("query '{}' answered in {:.2f}s".format(q, delta))
    stats = pd.Series(response_times).describe()
    print(stats)
    return stats


def run_load_test(rps=5, runtime=5):
    response_times = []
    threads = []
    print("initializing load test with %s requests/s" % (rps))
    t0 = time.time()
    for i in range(runtime * rps):
        def exec_requests(batch):
            q = random_query()
            r = requests.get(search_url, params={'q': q})
            delta = r.elapsed.total_seconds()
            response_times.append(delta)
            print("query {} was answered after {:.2f}s".format(batch , delta))

        t = Thread(target=exec_requests, args=(i+1,))
        t.start()
        threads.append(t)
        time.sleep(1.0 / rps)

    for t in threads:
        t.join()

    t1 = time.time()
    print("finished load test after {:.2f}s ({:.2f}s after last request)".format(t1-t0, t1-t0-runtime))
    stats = pd.Series(response_times).describe()
    print(stats)
    print("response times:", str(response_times))
    return stats


if __name__ == "__main__":
    run_queries_seq(num_queries=20, same_session=True)
    run_load_test(rps=1, runtime=60)
