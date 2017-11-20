import os
import sys
import uuid
import json
import pygal
import re
from jinja2 import Template
from Main import mainrun
from Config import MultiConfig

# How many iterations
ITERS = 5


def multirun(iter):
    # Create name for this multirun
    mname = str(uuid.uuid4())[:8]

    for ri in range(ITERS):

        mcf = MultiConfig()
        current_path = 'data/multirun_{mname}/{rint:0{width}d}/'.format(mname=mname, rint=ri,
                                                                        width=len(str(mcf.MAX_EPISODES)))
        os.makedirs(os.path.dirname(current_path), exist_ok=True)
        with open(current_path + 'config', 'w') as f:
            f.write(json.dumps(mcf.dump()))

        mainrun(mcf, fname=current_path + 'output', replay_path=os.path.abspath(current_path))

        chartdata = []
        meandata = [0, 0]
        with open(current_path + 'output', 'r') as f:
            for line in f:
                match = re.search('Episode: ([0-9]{0,6})\/[0-9]+, score: (-?[0-9]+)', line)
                if match:
                    chartdata.append((int(match.group(1)), int(match.group(2))))
                    meandata[0] = meandata[0] + int(match.group(2))
                    if int(match.group(1)) > meandata[1]:
                        meandata[1] = int(match.group(1))
        mean = meandata[0] / meandata[1]

        xychart = pygal.XY(stroke=False)
        xychart.title = 'Run {rint:0{width}d} - Mean: {mean}'.format(rint=ri, width=len(str(mcf.MAX_EPISODES)),
                                                                     mean=mean)
        xychart.add('Score', chartdata)
        xychart.render_to_file(current_path + 'graph.svg')

        with open(current_path + 'meta.json', 'w') as mf:
            mf.write(json.dumps({
                'path': '{rint:0{width}d}/graph.svg'.format(rint=ri, width=len(str(mcf.MAX_EPISODES))),
                'meanscore': mean,
                'runid': ri
            }))

    overviewbuilder(mname)


def overviewbuilder(mname):
    '''
    :param mname: String as either '342915e9' or 'data/multirun_342915e9/'
    :return: nothing on sucess, False on failure
    '''


    mname = mname.replace('data/multirun_', '')
    mname = mname.replace('/', '')

    print('Building overview for multirun: {0}'.format(mname))

    runpath = 'data/multirun_{mname}'.format(mname=mname)
    svgpaths = []
    for subdir, dirs, files in os.walk(runpath):
        for file in files:
            if file == 'meta.json':
                # Since meta.json is the last file to be created, we can assume this run is completed
                with open(os.path.join(subdir, file), 'r') as jf:
                    svgpaths.append(json.loads(jf.read()))
    if len(svgpaths) == 0:
        print('No files found. Are you sure you provided a proper multirun name or folder?')
        return False
    sortedsvgpaths = sorted(svgpaths, key=lambda k: k['meanscore'], reverse=True)

    with open('multiview.html','r') as tf:
        template_string = tf.read()

    template = Template(template_string)
    rendered_template = template.render(svgpaths=sortedsvgpaths, multirunid=mname)

    with open('data/multirun_{mname}/overview.html'.format(mname=mname), 'w') as of:
        of.write(rendered_template)

    print('Done building overview for multirun: {0}'.format(mname))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        overviewbuilder(sys.argv[1])
    else:
        multirun(ITERS)