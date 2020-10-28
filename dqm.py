#!/usr/bin/env python3
#
# Copyright 2019-2020 F. Ferri, Ph. Gras

#Python 2 and 3 compatibility
#Ref.: https://python-future.org/compatible_idioms.html
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

#End of Python 2 and 3 compatibility

#from wsgiref.simple_server import make_server
from gevent.pywsgi import WSGIServer
from cgi import parse_qs, escape
from email.utils import formatdate

import string
import glob
import os.path
import webbrowser
import re
import mimetypes
import urllib.parse as urlparse
import argparse
import sys
import copy
import html
import time

#Message display verbosity. Default is 1.
#Controlled with the -q (set to 0) and -v (increase by 1) options
verb = 1

root_dir = "./"
plot_dir = "store"
run_subdir = 'runs'
live_subdir = 'live'
plot_not_found = "store/plot_not_found.svg"
web_root = "/"
run_format = "%06d" #format of run directory names
css_file = root_dir + 'css/dqm.css'

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
        self.mimetypes = { ".css": "text/css", ".html": "text/html", ".htm": "text/html",  ".svg": "image/svg+xml", ".svgz": "image/svg+xml", "sgv.gz": "image/svg+xml" }
        self._default = self.dummy_gen

    def _get_mimetypes(self, path):
        #return mimetypes.guess_type(path)[0]
        return mimetypes.guess_type(path)

    def set_default(self, func):
        self.default = func

    def register(self, path, func):
        if path == "/" and self._default == self.dummy_gen:
            self._default = func
        msg(2, 'Registering %s -> %s()' %(path, func))
        self.map[path] = func

    def _is_static(self, path):
        if re.match("^/(css|static|store)/", path):
            return True
        else:
            return False

    def is_path_allowed(path):
        abs_root_dir = os.path.realpath(root_dir)
        path = os.path.realpath(path)

        if os.path.commonprefix((path, abs_root_dir)) != abs_root_dir:
            msg(1, "Request to access unsafe path %s rejected." % path)
            return False

        #Extra check, redundant with previous one, for safety:
        if ".." in path.split(os.path.sep):
            msg(1, "Bug Request to access unsafe path %s rejected." % path)
            return False

        return True

    def _gen_static(self, path):
        abspath = os.path.realpath(os.path.join(root_dir, path.lstrip('/')))
        if not PageStore.is_path_allowed(abspath):
            msg(1, "Ignoring request for file %s" % abspath)
            return ""
        try:
            with open(abspath, 'rb') as f:
                return f.read(max_static_page_size)
        except:
            e = sys.exc_info()[0]
            msg(1, "Failed to read file %s%s: %s" % (root_dir, path, e.msg))
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
            return None

        page = list(func(environ))

        if isinstance(page[1], str):
            page[1] = page[1].encode('utf-8')

        return page

store = PageStore()

def _look_for_img(base, exts):
    for ext in exts:
        img_path = "%s.%s" % (base, ext)
        if os.path.isfile(img_path) and not os.path.getsize(img_path) <= 50:
            return img_path
    return None

def get_plots(plot_dir):
    i = 0
    lres_exts = ["png", "gif", "jpg", "jpeg"]
    hres_exts = ["svg.gz", "svgz", "svg"]
    all_exts = list(set(lres_exts + hres_exts))
    paths = natural_sort(glob.iglob(plot_dir + "/*"))
    for p in paths:
        i += 1
        name = os.path.basename(p)
        msg(3, "Found directory %s" % p)
        file_found = False
        base = "%s/%s"  % (p, name)
        hres_img_path = _look_for_img(base, hres_exts)
        lres_img_path = _look_for_img(base, lres_exts)

        if lres_img_path:
            lres_img_path = add_rev(lres_img_path)

        if hres_img_path:
            hres_img_path = add_rev(hres_img_path)

        if lres_img_path and not hres_img_path:
            msg(1, "Missing high-resolution image")
            hres_img_path = lres_img_path

        if hres_img_path and not lres_img_path:
            msg(1, "Missing low-resolution image")
            lres_img_path = hres_img_path
                        
        if not lres_img_path:
            msg(1, "No image file %s/%s.xxx found. Supported extension: %s" % (p, name, " ".join(all_exts)))
            if os.path.isfile(plot_not_found):
                lres_img_path = plot_not_found
                hres_img_path = plot_not_found
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
        yield {"lres_img_path": lres_img_path, "hres_img_path": hres_img_path, "img_alt": img_alt, "caption_label": caption_label, "caption_text": caption_text }

    if i == 0:
        msg(1, "No plot found under directory %s" % plot_dir)


def add_rev(file_path):
    try:
        ts = int(os.stat(file_path).st_mtime)
        file_path += "!%d" % ts
    except Exception as e:
        print("Failed to add revision tag to file %s\n%s" % (file_path, str(e)))

    return file_path


def build_page(template_path, values):
    with open(template_path) as f:
        s = string.Template(f.read())

    return s.safe_substitute(values)

def gen_plot_page(plot_path):
    plots = get_plots(os.path.join(plot_dir, plot_path))
    figures = ""
    for p in plots:
        figures += build_page("templates/figure.thtml", p)

    return gen_page(content=figures, path=plot_path)

def gen_page(content = "&nbsp", content_class="flexcontent", path=""):
    try:
        css_ts = int(os.stat(css_file).st_mtime)
        css_version = "!%d" % css_ts
    except Exception as e:
        print(e)
        css_version = ""

    run_num = path_to_run_num(path)
    if run_num:
        run_num = run_format % run_num
    else:
        run_num = ""

    content = build_page("templates/main.thtml",
                         {"content": content, "content_class": content_class,
                          "path": html_path(path),
                          "run_subdir": run_subdir,
                          "live_subdir": live_subdir,
                          "run_num": run_num,
                          "css_version": css_version})

    return ("text/html", None), content

def gen_home(environ):
    s = open("templates/welcome_home.html").read()
    return gen_page(content=s)

store.register("/", gen_home)


#def gen_live(environ):
#    return gen_navigation(environ, is_live=True)

#store.register("/live", gen_live)

def gen_run(environ):
    params = urlparse.parse_qs(environ['QUERY_STRING'])
    if not params:
        return gen_navigation_(run_subdir)
    try:
        run_num = params["num"][0]
    except KeyError:
        msg(2, "Page run requested without passing a run number")
        return gen_live(environ)

    try:
        run_num = int(run_num)
    except ValueError:
        run_num = 0

    plot_path = plot_path_of_run(run_num)

    if not plot_path:
        plot_path = run_subdir

    return gen_navigation_(plot_path)

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




def path_to_run_num(plot_path):
    elts = plot_path.split(os.path.sep)
    for e in elts:
        try:
            run_num = int(e)
        except:
            continue

        if run_format % run_num == e:
            return run_num

    return None

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

    return gen_navigation_(plot_path)

def gen_navigation_(plot_path):
    dir_list = ""

    if ".." in plot_path.split(os.path.sep):
        msg(0, 'Ignoring navigation request with a path parameter containing "..".')
        return gen_page()

    charset = '[a-zA-Z0-9.\\-_/]'
    if not re.match('^' + charset + '*$', plot_path):
        msg(0, 'Ignoring navigation request to path %s that contains other character(s) than from the %s set.' % (plot_path, charset))
        return gen_page()
    
    d = os.path.join(plot_dir, plot_path.lstrip('/'))
    
    for r in get_dir_list(d):
        new_path = os.path.join(plot_path, r)
        if is_plot_dir(os.path.join(plot_dir, new_path)):
            return gen_plot_page(plot_path)
        else:
            dir_list += build_page("templates/run.thtml", {"link": "nav?path=%s" % urlparse.quote(new_path), "run_num": r})

    return gen_page(content=dir_list, content_class="runlist", path=plot_path)

store.register("/nav", gen_navigation)
#store.register("/runs", gen_navigation)

def resubs(string, substitutions):
    '''Apply a list of regex substitions. The substitutions argument must a be a list of pairs (pattern, to_sustitute).'''
    new_str = string
    for s in substitutions:
        new_str = re.sub(s[0], s[1], new_str)
    return new_str

def html_path(path):
    '''Encode plot navigation PATH in html and includes links.'''
    dirs = path.split('/')
    result = ""
    path = ""
    sep = ""
    for d in dirs:
        path = os.path.join(path, d)
        result += sep + '<a href="nav?path=%s">%s</a>' % (urlparse.quote(path), html.escape(d))
        sep = "&nbsp;&gt; "

    return result

def plot_path_of_run(run_num):
    try:
        run_num = int(run_num)
    except ValueError:
        run_num = 0

    run_label = run_format % run_num

    #Returns first found directory:
    for i in glob.iglob(os.path.join(plot_dir, run_subdir, '**', run_label, ''), recursive=True):
        return os.path.relpath(os.path.abspath(i), start = os.path.abspath(plot_dir))
    return None

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


    path = environ.get('PATH_INFO')

    elts = path.rsplit('!', 1)

    if len(elts) > 1:
        revision = elts[1]
    else:
        revision = None

    #Drop revision number from page path for further processing
    environ['PATH_INFO'] = elts[0]

    resp = store.gen_page(environ)

    if not resp:
        start_response('404 Not Found', [('Content-Type', 'text/html')])
        return [b'<h1>Not Found</h1>']

    mime_type, response_body = resp

    status = '200 OK'

    # Now content type is text/html
    response_headers = [
        ('Content-Type', mime_type[0]),
        ('Content-Length', str(len(response_body)))
    ]

    if mime_type[1]:
        response_headers.append(('Content-Encoding', mime_type[1]))

    #Enable browser cache If requested content supports revision
    if revision:
        one_month = 3600.*24.*30.
        expire_date = time.time() + one_month
        response_headers.append(('Expires', formatdate(expire_date, localtime=False, usegmt=True)))

    start_response(status, response_headers)
    return [response_body]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cupid/Cross data quality monitoring web server')
    parser.add_argument('--host', '-H', default='192.168.3.152', help='Specifies host.')
    parser.add_argument('--port', '-p', default=8787, type=int, help='Specifies port to listen to.')
    parser.add_argument('--verbose', '-v', action='count', default=1, help='Increase verbosity')
    parser.add_argument('--quiet', '-q', action='count', default=1, help='Quiet mode. Reduces message at the minium.')
    opt = parser.parse_args()
    verb = opt.verbose

    if opt.quiet:
        verb = 0

    http_server = WSGIServer((opt.host, opt.port), application)
    http_server.serve_forever()

    msg(1, "Listening to http://%s:%d" % (opt.host, opt.port))

#    webbrowser.open("http://localhost:8051")

    # Now it is serve_forever() in instead of handle_request()
    httpd.serve_forever()
