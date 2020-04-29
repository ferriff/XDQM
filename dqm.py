#!/usr/bin/env python

#Python 2 and 3 compatibility
#Ref.: https://python-future.org/compatible_idioms.html
from __future__ import print_function 

#End of Python 2 and 3 compatibility

from wsgiref.simple_server import make_server
from cgi import parse_qs, escape

import string
import glob
import os.path
import webbrowser
import re
import mimetypes
import urlparse
#import urllib.parse

verb = 1

root_dir = "./"
plot_dir_live = "store/live"
plot_dir_runs = "store/runs"
plot_not_found = "store/plot_not_found.svg"
web_root = "/"

max_caption_file_size = 1024*1024
max_static_page_size = 100*1024*1024

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def msg(level, *args):
    if verb >= level:
        print(*args)

class PageStore(object):
    @staticmethod
    def dummy_gen(environ):
        return ("text", "")
    
    def __init__(self):
        self.map = {}
        self.mimetypes = { ".css": "text/css", ".html": "text/html", ".htm": "text/html",  }
        self._default = self.dummy_gen

    def _get_mimetypes(self, path):
        #return mimetypes.guess_type(path)[0]
        return mimetypes.guess_type(path)

    def set_default(self, func):
        self.default = func
        
    def register(self, path, func):
        if path == "/" and self._default == self.dummy_gen:
            self._default = func
        print('Registering %s -> %s()' %(path, func))
        self.map[path] = func

    def _is_static(self, path):
        if re.match("^/(css|static|store)/", path):
            return True
        else:
            return False

    def is_path_allowed(self, path):
        if os.path.commonprefix((root_dir + os.path.realpath(path), root_dir)) != root_dir:
            msg(1, "Request to access unsafe path %s rejected." % path)
            return False
        else:
            return True
        
    def _gen_static(self, path):
        if not self.is_path_allowed(path):
            return ""
        try:
            with open(root_dir + path) as f:
                return f.read(max_static_page_size)
        except:
            msg(1, "Failed to read file %s%s" % (root_dir, path))
            return ""
        
    def gen_page(self, environ):
        path = environ.get('PATH_INFO')
        msg(2, "Page %s requested" % path)

        if self._is_static(path):
            mimetype = self._get_mimetypes(path)
            if not mimetype:
                msg(1, "Extension of file %s is not supported." % path)
                return self._default(environ)                
            else:
                msg(3, "Serving static page %s as %s type." % (path, mimetype) )
                return mimetype, self._gen_static(path)

        func = self._default
        try:
            func = self.map[path]
        except KeyError:
            pass

        return func(environ)

store = PageStore()

def get_plots(plot_dir):
    i = 0
    exts = ["svg.gz", "svgz", "svg", "png", "gif", "jpg", "jpeg"]
    paths = natural_sort(glob.iglob(plot_dir + "/*"))
    for p in paths:
        i += 1
        name = os.path.basename(p)
        msg(3, "Found directory %s" % p)
        file_found = False
        for ext in exts:
            img_path = "%s/%s.%s" % (p, name, ext)
            if os.path.isfile(img_path) and not os.path.getsize(img_path) <= 50:
                file_found = True
                break

        if not file_found:
            msg(1, "File %s was not found" % img_path)
            if os.path.isfile(plot_not_found):
                img_path = plot_not_found
            else:
                continue
        
        caption_file = "%s/%s.cap" % (p, name)
        try:
            with open(caption_file) as f:
                caption_text = f.read(max_caption_file_size);
        except IOError:
            caption_text = name

        caption_label = "Plot %d" % i
        img_alt = caption_label
        yield {"img_path": img_path, "img_alt": img_alt, "caption_label": caption_label, "caption_text": caption_text }
        
    if i == 0:
        msg(1, "No plot found under directory %s" % plot_dir)

def build_page(template_path, values):
    with open(template_path) as f:
        s = string.Template(f.read())

    return s.substitute(values)

def gen_plot_page(environ, plot_dir):
    plots = get_plots(plot_dir)
    figures = ""
    for p in plots:
        figures += build_page("templates/figure.thtml", p)

    home = build_page("templates/main.thtml", { "content": figures, "content_class": "flexcontent"})

    return "text/html", home

def gen_home(environ):
    return "text/html", build_page("templates/main.thtml", {"content": "&nbsp;", "content_class": "flexcontent"})

store.register("/", gen_home)

def gen_live(environ):
    return gen_plot_page(environ, plot_dir_live)

store.register("/live", gen_live)

def gen_run(environ):
    print('>>>', environ)
    params = urlparse.parse_qs(environ['QUERY_STRING'])
    if not params:
        return gen_live(environ)
    try:
        run_num = params["num"][0]
    except KeyError:
        msg(2, "Page run requested without passing a run number")
        return gen_live(environ)
    
    return gen_plot_page(environ, os.path.join(plot_dir_runs, run_num))

store.register("/run", gen_run)


def get_run_list():
    r = []

    for p in glob.iglob(os.path.join(plot_dir_runs, "*")):
        #if not re.match(r'.*/\d+$', p) or not os.path.isdir(p): # FIXME: this is the original version
        #    continue
        #r.append(os.path.basename(p))
        if not re.match(r'.*/\d+$', p) and not os.path.isdir(p):
            continue
        r.append(os.path.basename(p))

        #if re.match(r'.*/\d+$', p):
        #    r.append([os.path.basename(p), 'r'])
        #elif os.path.isdir(p):
        #    r.append([os.path.basename(p), 'd'])
        #else:
        #    continue

    if(len(r) == 0):
        msg(1, "No run found!")
    r = natural_sort(r)
    for x in r:
        yield x


def gen_runs(environ):
    run_list = ""
    for r in get_run_list():
        run_list += build_page("templates/run.thtml", {"link": "run?num=%s" % r , "run_num": r})

    return "text/html", build_page("templates/main.thtml", {"content": run_list, "content_class": "runlist"})

#store.register("/runs", gen_runs)

def get_dir_list(path):
    d = []
    #for p in glob.iglob(os.path.join(plot_dir_runs, "*")):
    for p in glob.iglob(os.path.join(path, "*")):
        #if not re.match(r'.*/\d+$', p) or not os.path.isdir(p): # FIXME: this is the original version
        #    continue
        #r.append(os.path.basename(p))
        if not re.match(r'.*/\d+$', p) and not os.path.isdir(p):
            continue
        d.append(os.path.basename(p))

    if(len(d) == 0):
        msg(1, "No dir found!")
    d = natural_sort(d)
    for x in d:
        yield x

def gen_plot_page_simple(environ):
    path = environ.get('PATH_INFO')
    ppath = os.path.join('store/', path.lstrip('/'))
    return gen_plot_page(environ, ppath)

def is_plot_dir(path):
    name = os.path.basename(path)
    return len(glob.glob(os.path.join(path, name + '.*'))) > 0

def gen_navigation(environ):

    params = urlparse.parse_qs(environ['QUERY_STRING'])
    plot_path = ""
    if params:
        try:
            plot_path = params["path"][0]
        except KeyError:
            msg(2, "Page run requested without passing a path")
            pass

    dir_list = ""
    for r in get_dir_list(os.path.join(plot_dir_runs, plot_path.lstrip('/'))):
        new_path = os.path.join(plot_path, r)
        if is_plot_dir(os.path.join(plot_dir_runs, new_path)):
            return gen_plot_page(environ, os.path.join(plot_dir_runs, plot_path))
        dir_list += build_page("templates/run.thtml", {"link": "nav?path=%s" % new_path, "run_num": r})

    return "text/html", build_page("templates/main.thtml", {"content": dir_list, "content_class": "runlist"})

store.register("/nav", gen_navigation)
store.register("/runs", gen_navigation)



def application (environ, start_response):
##    # Returns a dictionary in which the values are lists
##    d = parse_qs(environ['QUERY_STRING'])
##
##    # As there can be more than one value for a variable then
##    # a list is provided as a default value.
##    age = d.get('age', [''])[0] # Returns the first age value
##    hobbies = d.get('hobbies', []) # Returns a list of hobbies
##
##    # Always escape user input to avoid script injection
##    age = escape(age)
##    hobbies = [escape(hobby) for hobby in hobbies]
##
##    response_body = html % { # Fill the above html template in
##        'checked-software': ('', 'checked')['software' in hobbies],
##        'checked-tunning': ('', 'checked')['tunning' in hobbies],
##        'age': age or 'Empty',
##        'hobbies': ', '.join(hobbies or ['No Hobbies?'])
##    }
##

    msg(9, "Hello")
    
    mime_type, response_body = store.gen_page(environ)

    msg(9, "Returning page of type %s:\n%s" % (mime_type, response_body))

#    mime_type, response_body = ("text", "Hello!")

    status = '200 OK'

    # Now content type is text/html
    response_headers = [
        ('Content-Type', mime_type[0]),
        ('Content-Length', str(len(response_body)))
    ]

    if mime_type[1]:
        response_headers.append(('Content-Encoding', mime_type[1]))

    start_response(status, response_headers)
    return [response_body]

    
    

if __name__ == "__main__":

    #httpd = make_server('localhost', 8051, application)

    #host, port = '132.166.9.40', 8051
    #host, port = '192.168.3.152', 443
    host, port = 'localhost', 8787
    httpd = make_server(host, port, application)

    print("Listening to http://%s:%d" % (host, port))

#    webbrowser.open("http://localhost:8051")
    
    # Now it is serve_forever() in instead of handle_request()
    httpd.serve_forever()
