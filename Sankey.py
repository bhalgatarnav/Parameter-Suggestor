import plotly.graph_objects as go

# Define all labels in order (nodes)
labels = [
    'Traditional High Street','Fast Fashion Model','Seasonal Collections',
    'Mass Production','Transactional Focus',
    'Sustainability Crisis','Digital Disruption','Post‑COVID Changes',
    'Changing Values','Tech Innovation',
    'Experiential Retail','Circular Economy','Hyper‑Personalization',
    'Community Hubs','AR/VR Shopping','Resale & Rental',
    'Slow Luxury','Customer Co‑creation','Omnichannel Experience'
]

# Create mapping from ID to index
label_index = {label: idx for idx, label in enumerate(labels)}

# Define the links (source, target, value)
links_raw = [
    ('Traditional High Street','Sustainability Crisis',25),
    ('Fast Fashion Model','Sustainability Crisis',30),
    ('Traditional High Street','Digital Disruption',35),
    ('Seasonal Collections','Changing Values',20),
    ('Mass Production','Sustainability Crisis',40),
    ('Transactional Focus','Post‑COVID Changes',30),
    ('Transactional Focus','Changing Values',25),
    ('Sustainability Crisis','Circular Economy',45),
    ('Sustainability Crisis','Slow Luxury',35),
    ('Sustainability Crisis','Resale & Rental',25),
    ('Digital Disruption','AR/VR Shopping',30),
    ('Digital Disruption','Omnichannel Experience',40),
    ('Digital Disruption','Hyper‑Personalization',35),
    ('Post‑COVID Changes','Experiential Retail',40),
    ('Post‑COVID Changes','Community Hubs',35),
    ('Post‑COVID Changes','Omnichannel Experience',25),
    ('Changing Values','Experiential Retail',35),
    ('Changing Values','Community Hubs',30),
    ('Changing Values','Customer Co‑creation',25),
    ('Tech Innovation','AR/VR Shopping',40),
    ('Tech Innovation','Hyper‑Personalization',45),
    ('Tech Innovation','Omnichannel Experience',30)
]

sources = [label_index[src] for src,_,_ in links_raw]
targets = [label_index[tgt] for _,tgt,_ in links_raw]
values  = [val for _,_,val in links_raw]

# Optional: color categories
node_colors = ['#ff6b6b']*5 + ['#4ecdc4']*5 + ['#45b7d1']*9

fig = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=labels,
        color=node_colors
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=['rgba(70,130,180,0.4)']*len(sources)  # uniform link color
    )
)])

fig.update_layout(
    title_text="Fashion Retail Evolution: Emerging Trends Flow",
    font=dict(size=12, family="Arial"),
    width=1000,
    height=600
)

fig.show()
fig.write_html("fashion_retail_sankey.html")

