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

root_dir = "."
extra_root_dirs=["/home/dqm/dqm/xdqm-devel"]
plot_dir = "store/runs"
run_subdir = 'DATA'
cryo_subdir = 'cryo'
live_subdir = ''
plot_not_found = "store/plot_not_found.png"
web_root = "/"
run_num_format = "%06d" #format of run directory names
run_num_format_regex = re.compile(r"\d{6}")
chunk_num_format = "%03d" #format of run chunk directory names
chunk_num_format_regex = re.compile(r"\d{3}")
css_file = os.path.join(root_dir, 'css/dqm.css')

max_caption_file_size = 1024*1024
max_static_page_size = 100*1024*1024

# Priorities to modify order of run and cryo-day folders when listed
# 0: no modification
# >0: push to top of the list, highest value first
# <0: push to top of the list, most negative last
dir_priorities = {"Now": 1}

def n_subdirs(dir):
    '''Return the number of subdirectory the directory dir contains'''
    n = 0
    for subdir in os.listdir(dir):
        if os.isdir(os.join(dir, subdir)):
            n += 1

    return n

def natural_sort(l, reverse = False):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key, reverse = reverse)

def msg(level, *args):
    if verb >= level:
        print(*args, "(%s)" % level)

class PageStore(object):
    @staticmethod
    def dummy_gen(environ):
        return ("text", "")

    def __init__(self):
        self.map = {}
        self.mimetypes = { ".css": "text/css", ".html": "text/html", ".htm": "text/html",  ".svg": "image/svg+xml", ".svgz": "image/svg+xml", "sgv.gz": "image/svg+xml", "png": "image/png" }
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
        path = os.path.realpath(path)

        passed = False

        for d in [root_dir, *extra_root_dirs]:
            abs_root_dir = os.path.realpath(d)
            if os.path.commonprefix((path, abs_root_dir)) == abs_root_dir:
                passed |= True
                break

        if not passed:
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
        if os.path.isfile(img_path) and os.path.getsize(img_path) > 50:
            return img_path
    return None

def get_plots(plot_dir, ref_plot_dir = None):
    '''Retrieves the paths to the plots stored in the plot_dir directory.
    If the parameter ref_plot_dir is different than None, then for
    each plot, a plot with the same file name (extension excluded) is
    searched for in this ref_plot_dir redirectory. If no plot is
    found, it is substitueted by a placeholder image.

    This method is a generator that yields information for plot at
    each call iteration. The information includes: two image file
    paths, one for low resolution and one for high resolution image; a
    text for the alt attribute of html anchor tag; plot caption label
    and text; two image file paths for the reference plot if
    ref_plot_dir is not None.

    Returns an dict with keys lres_img_path, hres_img_path, img_alt,
    caption_label, caption_text, lres_ref_img_path and
    hres_ref_img_path. The last two are not included if ref_plot_dir
    parameter value is None.

    '''

    i = 0
    lres_exts = ["png", "gif", "jpg", "jpeg"]
    hres_exts = ["svg.gz", "svgz", "svg"]
    all_exts = list(set(lres_exts + hres_exts))
    paths = natural_sort(glob.iglob(plot_dir + "/*"))

    if ref_plot_dir and not os.path.isdir(ref_plot_dir):
        msg(2, "Reference plot directory '%s' was not found" % ref_plot_dir)
        ref_plot_dir = None

    for p in paths:
        i += 1
        name = os.path.basename(p)
        msg(3, "Found directory %s" % p)
        file_found = False
        base = "%s/%s"  % (p, name)
        hres_img_path = _look_for_img(base, hres_exts)
        lres_img_path = _look_for_img(base, lres_exts)

        if ref_plot_dir:
            ref_base = re.sub(r'^' + plot_dir.rstrip(os.path.sep) + os.path.sep, ref_plot_dir.rstrip(os.path.sep) + os.path.sep, base)
            lres_ref_img_path = add_rev(_look_for_img(ref_base, lres_exts))
            hres_ref_img_path = add_rev(_look_for_img(ref_base, hres_exts))

        hres_img_path = add_rev(hres_img_path)
        lres_img_path = add_rev(lres_img_path)

        if lres_img_path and not hres_img_path:
            msg(1, "Missing high-resolution image")
            hres_img_path = lres_img_path

        if hres_img_path and not lres_img_path:
            msg(1, "Missing low-resolution image")
            lres_img_path = hres_img_path

        if ref_plot_dir:
            if lres_ref_img_path and not hres_ref_img_path:
                msg(1, "Missing high-resolution image for the reference plot")
                hres_ref_img_path = lres_ref_img_path

            if hres_ref_img_path and not lres_img_path:
                msg(1, "Missing low-resolution image for the reference plot")
                lres_ref_img_path = hres_ref_img_path

            
            if not lres_ref_img_path:
                msg(1, "No image file %s/%s.xxx found. Supported extension: %s" % (ref_plot_dir, name, " ".join(all_exts)))
                if os.path.isfile(plot_not_found):
                    lres_ref_img_path = plot_not_found
                    hres_ref_img_path = plot_not_found
                else:
                    lres_ref_img_path = None
                    hres_ref_img_path = None

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
        res =  {"lres_img_path": lres_img_path, "hres_img_path": hres_img_path, "img_alt": img_alt, "caption_label": caption_label, "caption_text": caption_text }
        if ref_plot_dir:
            res["caption_label"] += "a"
            res["ref_caption_label"] = caption_label + "b"
            res["lres_ref_img_path"] = lres_ref_img_path
            res["hres_ref_img_path"] = lres_ref_img_path
        yield res

    if i == 0:
        msg(1, "No plot found under directory %s" % plot_dir)


def add_rev(file_path):
    if file_path is None:
        return None
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

def gen_plot_page(plot_path, compare_on, ref_run_chunk):

    if compare_on:
        ref_plot_path = change_run_and_chunk_in_path(plot_path, *ref_run_chunk)
    else:
        ref_plot_path = None
    
    plots = get_plots(os.path.join(plot_dir, plot_path), os.path.join(plot_dir, ref_plot_path) if ref_plot_path is not None  else None)

    figures = ""
    for p in plots:
        figures += build_page("templates/figure.thtml", p)
        if ref_plot_path:
            p_ref = p
            p_ref["hres_img_path"] = p_ref["hres_ref_img_path"]
            p_ref["lres_img_path"] = p_ref["lres_ref_img_path"]
            p_ref["caption_label"] = p_ref["ref_caption_label"]
            figures += build_page("templates/figure.thtml", p)

    return gen_page(content=figures, path=plot_path, root = "Runs", compare_on = compare_on, ref_run_chunk = ref_run_chunk)

def run_chunk_dir(run, chunk):
    return os.path.join(run_num_format % run, chunk_num_format % chunk)

def run_chunk_label(run, chunk):
    if not isinstance(run, int):
        return ""
    label = run_num_format % run
    if isinstance(chunk, int):
        label += "." + chunk_num_format % chunk

    return label

def gen_page(content = "&nbsp", content_class="flexcontent", path="", root="Home", compare_on = False, ref_run_chunk = [0, 0]):
    try:
        css_ts = int(os.stat(css_file).st_mtime)
        css_version = "!%d" % css_ts
    except Exception as e:
        print(e)
        css_version = ""

    run_num, chunk_num, _ = path_to_run_chunk(path)

    if run_num:
        run_chunk = run_chunk_label(run_num, chunk_num)
    else:
        rc = last_run_chunk()
        run_chunk = run_chunk_label(*rc) if rc else ""

    ref_run_chunk_tag = run_chunk_label(*ref_run_chunk)
        
    content = build_page("templates/main.thtml",
                         {"content": content, "content_class": content_class,
                          "path_nav": html_path(root, path),
                          "path": path,
                          "run_subdir": run_subdir,
                          "cryo_subdir": cryo_subdir,
                          "live_subdir": live_subdir,
                          "run_chunk": run_chunk,
                          "ref_run_chunk": ref_run_chunk_tag,
                          "css_version": css_version,
                          "compare_checked": "checked" if compare_on else"",
                          "compare_disabled": "" if compare_on else "disabled",
                         })

    return ("text/html", None), content

def gen_home(environ):
    s = open("templates/welcome_home.html").read()
    return gen_page(content=s)



def gen_run(environ):

    params = urlparse.parse_qs(environ['QUERY_STRING'])

    if not params:
        return gen_navigation_(run_subdir, False, [0,0])
    try:
        run_chunk = params["run_chunk"][0]
    except KeyError:
        run_chunk = ''

    run_chunk_elts = run_chunk.split(".")

    try:
        run_num = int(run_chunk_elts[0])
    except ValueError:
        run_num = 0

    if run_num <= 0:
        tmp_ = last_run_chunk()
        if tmp_:
            run_num, chunk_num = tmp_
            if chunk_num is None:
                chunk_num = 0
    else:
        try:
            chunk_num = int(run_chunk_elts[1])
        except (IndexError, ValueError):
            chunk_num = None
            
    try:
        old_path = params["path"][0]
    except KeyError: #will happend if path is empty
        old_path = ''

    plot_path =  change_run_and_chunk_in_path(old_path, run_num, chunk_num)
    if plot_path is None:
        plot_path = ""

    compare_on, ref_run_chunk = get_compare_params(params)
    return gen_navigation_(plot_path, compare_on, ref_run_chunk)


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




def path_to_run_chunk(plot_path):
    elts = plot_path.split(os.path.sep)
    run_num, chunk_num, i = None, None, None
    for i,e in enumerate(elts):
        try:
            run_num = int(e)
            if i + 1 < len(elts) :
                chunk_num = int(elts[i+1])
            else:
                chunk_num = None
        except:
            continue

        if (run_num_format % run_num) != e:
            run_num = None
            i = None

        if chunk_num and (chunk_num_format % chunk_num) != elts[i+1]:
            chunk_num = None

        break

    return run_num, chunk_num, i

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
        msg(1, "No dir found at path '%s'!" % path)

    #In order to have first most recent runs and chunks on top of the
    #list, we move directories named as numbers first ordered by
    #decreasing order, then we put the alphanumeric ones
    #in alphabetical oders.
    d_num = []
    d_alphanum = []
    for x in d:
        if re.match(r"^\d+$", x):
            d_num.append(x)
        else:
            d_alphanum.append(x)

    d = natural_sort(d_num, reverse=True) + natural_sort(d_alphanum, reverse=False)

    d.sort(key = lambda x: dir_priorities[x] if x in dir_priorities else 0, reverse = True)
    
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

    compare_on, ref_run_chunk = get_compare_params(params)
    
    return gen_navigation_(plot_path, compare_on, ref_run_chunk)

def get_compare_params(params):
    if "compare" in params: # and len(params['compare'][0])>0:
        compare_on = True
    else:
        compare_on = False

    if "ref_run_chunk" in params:
        ref_run_chunk = get_run_chunk(params["ref_run_chunk"][0])
    else:
        ref_run_chunk = get_run_chunk("")

    return compare_on, ref_run_chunk

def get_run_chunk(run_chunk_tag):
        run_chunk_elts = run_chunk_tag.split(".")
        try:
            run_num = int(run_chunk_elts[0])
        except ValueError:
            run_num = 0

        try:
            chunk_num = int(run_chunk_elts[1])
        except (IndexError, ValueError):
            chunk_num = 0

        return [run_num, chunk_num]

def run_dir_filter(iterable):
    '''Selects run directory strings from the passed iterable.
       This function is a generator'''
    for name in iterable:
        if isinstance(name, str)  and run_num_format_regex.match(name):
            yield name

def chunk_dir_filter(iterable):
    '''Selects chunk directory strings from the passed iterable.
       This function is a generator'''
    for name in iterable:
        if isinstance(name, str)  and chunk_num_format_regex.match(name):
            yield name    

def first_chunk_of_run(run):
    try:
        chunk = min(chunk_dir_filter(glob.iglob(os.path.join(plot_dir, run_subdir, run_num_format % run))))
    except ValueError: #expected if the list is empty
        return None        
            
def last_run_chunk():
    try:
        run = max(run_dir_filter(map(os.path.basename, glob.iglob(os.path.join(plot_dir, run_subdir, '*')))))
    except ValueError: #expected if the list is empty
        return None

    try:
        chunk = max(chunk_dir_filter(glob.iglob(os.path.join(plot_dir, run_subdir, run))))
    except ValueError: #expected if the list is empty
        return [int(run.lstrip('0')), None]

    #FIXME: the code below assumes a specific format for run and chunk dirs
    return [int(run.lstrip('0')), int(chunk.lstrip('0'))]
                                     
    
    
def gen_navigation_(plot_path, compare_on, ref_run_chunk):
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
            return gen_plot_page(plot_path, compare_on, ref_run_chunk)
        else:
            if r == "Now":
                r = "&nbsp;" * 3 + r + "&nbsp;"*2
            dir_list += build_page("templates/run.thtml", {"link": "?page=nav&path=%s" % urlparse.quote(new_path), "run_num": r})



    return gen_page(content=dir_list, content_class="runlist", path=plot_path, root = "Runs")


def gen_dispatcher(environ):
    params = urlparse.parse_qs(environ['QUERY_STRING'])

    try:
        page = params["page"][0]
    except (KeyError, TypeError, IndexError):
        page = 'home'

    if page == 'home':
        return gen_home(environ)
    elif page == 'run':
        return gen_run(environ)
    elif page == 'nav':
        return gen_navigation(environ)

    
store.register("/", gen_dispatcher)

def resubs(string, substitutions):
    '''Apply a list of regex substitions. The substitutions argument must a be a list of pairs (pattern, to_sustitute).'''
    new_str = string
    for s in substitutions:
        new_str = re.sub(s[0], s[1], new_str)
    return new_str

def html_path(root, path):
    '''Encode plot navigation PATH in html and includes links.'''
    dirs = path.split('/')
    nskip = 0
    if (root == "Runs") and (os.path.commonprefix((path, cryo_subdir)) == cryo_subdir):
        root = "Cryo"

    if root == "Home":
        result = '<a href="/">%s</a>' % html.escape(root)
    elif root == "Runs":
        result = '<a href="?page=nav&path=%s">%s</a>' % (urlparse.quote(run_subdir), html.escape(root))
        nskip = len([x for x in os.path.split(run_subdir) if x])
    elif root == "Cryo":
        result = '<a href="?page=nav&path=%s">%s</a>' % (urlparse.quote(cryo_subdir), html.escape(root))
        nskip = len([x for x in os.path.split(cryo_subdir) if x])
    else:
        result = html.escape(root)

    path = ""
    sep = "&nbsp;&gt; "
    skipped = 0
    for d in dirs:
        if len(d.strip())==0:
            continue
        path = os.path.join(path, d)
        if skipped < nskip:
            skipped += 1
        else:
            result += sep + '<a href="?page=nav&path=%s">%s</a>' % (urlparse.quote(path), html.escape(d))

    return result

def change_run_and_chunk_in_path(path, run_num, chunk_num):
    old_run_num, old_chunk_num, run_num_pos = path_to_run_chunk(path)
    if not old_run_num:
        msg(2, "Could not find run numnber in path '%s'." % path)
        if chunk_num is None:
            return os.path.join(run_subdir, run_num_format % run_num)
        else:
            return os.path.join(run_subdir, run_chunk_dir(run_num, chunk_num))
        
    chunk_num_pos = run_num_pos + 1        
    dirs = path.split(os.path.sep)
    dirs[run_num_pos] = run_num_format % run_num
    
    if chunk_num is None:
        chunk_num = first_chunk_of_run(run_num)
        if chunk_num is None:
            chunk_num = 0
        chunk_num_not_specified = True
    else:
        chunk_num_not_specified = False
        
    try:
        dirs[chunk_num_pos] = chunk_num_format % chunk_num
    except IndexError:
        msg(2, "No chunk directory found in path '%s'." % path)
        pass

    #ensure return path exists.
    #If it doesnt(e.g. requested chunk or run does not exist),
    #we go up to one directory until we find an existing directory:
    i = len(dirs)
    while i > 0:
        path = os.path.join(*dirs[:i])
        if os.path.isdir(os.path.join(plot_dir, path)):
            break
        i -= 1

    #if we went up to the chunk directory, the chunk num was not specified, and the run
    #contains several chunks we will return the run directory instead of chunk 000 directory:
    if (i - 1) == chunk_num_pos and chunk_num_not_specified and nsubdirs(os.join(*dirs[:i])) > 1:
        i =- 1

    return os.path.join(*dirs[:i]) if i > 0 else None
        
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
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet mode. Reduces message at the minium.')
    opt = parser.parse_args()
    verb = opt.verbose

    if opt.quiet:
        verb = 0

    http_server = WSGIServer((opt.host, opt.port), application, log=None)
    http_server.serve_forever()

    msg(1, "Listening to http://%s:%d" % (opt.host, opt.port))

#    webbrowser.open("http://localhost:8051")

    # Now it is serve_forever() in instead of handle_request()
    httpd.serve_forever()
