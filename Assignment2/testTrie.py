import pygtrie

t = pygtrie.StringTrie()
t['foo'] = 'Foo'
t['foo/bar'] = 'Bar'
t['foo/bar/baz/qux'] = 'Baz'

# del t['foo/bar']
print(t.has_key('Bar'))
# del t['foo':]
# print(t.keys())
