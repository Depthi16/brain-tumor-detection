import os

files = ['templates/history.html', 'templates/performance.html']
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    if '        </div>\n        </div>\n    </div>' in content:
        content = content.replace('        </div>\n        </div>\n    </div>', '        </div>\n    </div>', 1)
    elif '    </div>\n    </div>\n</div>' in content:
        content = content.replace('    </div>\n    </div>\n</div>', '    </div>\n</div>', 1)
        
    if '`n        <a href="/history">History</a>' in content:
        content = content.replace('`n        <a href="/history">History</a>', '\n        <a href="/history">History</a>')
        
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
