import os
import uuid
import json
import pygal
import re
from jinja2 import Template
from Main import mainrun
from Config import MultiConfig

# How many iterations
ITERS = 30


# Create name for this multirun
mname = str(uuid.uuid4())[:8]

svgpaths = []

for ri in range(ITERS):

    mcf = MultiConfig()
    current_path = 'data/multirun_{mname}/{rint:0{width}d}/'.format(mname=mname, rint=ri, width=len(str(mcf.MAX_EPISODES)))

    mainrun(mcf, fname=current_path + 'output', replay_path=os.path.abspath(current_path))

    with open(current_path + 'config', 'w') as f:
        f.write(json.dumps(mcf.dump()))

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
    xychart.title = 'Run {rint:0{width}d} - Mean: {mean}'.format(rint=ri, width=len(str(mcf.MAX_EPISODES)), mean=mean)
    xychart.add('Score', chartdata)
    xychart.render_to_file(current_path + 'graph.svg')

    #Add to svgpaths in order to render template later
    svgpaths.append({
        'path': '{rint:0{width}d}/graph.svg'.format(rint=ri, width=len(str(mcf.MAX_EPISODES))),
        'meanscore': mean,
        'runid': ri
    })

# Multirun is now over and we do datasorting and template building.

sortedsvgpaths = sorted(svgpaths, key=lambda k: k['meanscore'], reverse=True)

template_string = ''
with open('multiview.html','r') as tf:
    template_string = tf.read()

template = Template(template_string)
rendered_template = template.render(svgpaths=sortedsvgpaths, multirunid=mname)

with open('data/multirun_{mname}/overview.html'.format(mname=mname), 'w') as of:
    of.write(rendered_template)

print('Done with multirun: {0}'.format(mname))
