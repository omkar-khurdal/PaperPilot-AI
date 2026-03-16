.....













# autocpmplete engine







class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Autocomplete:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def dfs(self, node, prefix, results, k):
        if len(results) >= k:
            return

        if node.is_end:
            results.append(prefix)

        for char in node.children:
            self.dfs(node.children[char], prefix + char, results, k)

    def autocomplete(self, prefix, k=3):
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []

            node = node.children[char]

        results = []
        self.dfs(node, prefix, results, k)
        return results


# Dictionary
words = ["apple", "application", "apply", "appetite", "banana"]

engine = Autocomplete()

for w in words:
    engine.insert(w)

query = input("Search: ")

print("Top suggestions:", engine.autocomplete(query, 3))



# flask


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Dictionary words
words = [
    "apple",
    "application",
    "apply",
    "appetite",
    "banana",
    "bat",
    "battle",
    "cat",
    "cater",
    "dog"
]


# Main page
@app.route("/")
def home():
    return render_template("index.html")


# Autocomplete API
@app.route("/search")
def search():
    query = request.args.get("q", "").lower()

    suggestions = [w for w in words if w.startswith(query)]

    return jsonify(suggestions[:5])  # return top 5


if __name__ == "__main__":
    app.run(debug=True)
# html index

<!DOCTYPE html>
<html>
<head>

<title>Autocomplete Search</title>

<style>

body{
font-family: Arial;
margin:50px;
}

#searchBox{
width:300px;
padding:10px;
font-size:16px;
}

#suggestions{
border:1px solid #ccc;
width:320px;
}

.item{
padding:8px;
cursor:pointer;
}

.item:hover{
background:#f0f0f0;
}

</style>

</head>

<body>

<h2>Autocomplete Search Engine</h2>

<input type="text" id="searchBox" placeholder="Type something...">

<div id="suggestions"></div>


<script>

const searchBox = document.getElementById("searchBox");
const suggestions = document.getElementById("suggestions");


searchBox.addEventListener("input", function(){

let query = searchBox.value;

fetch(`/search?q=${query}`)
.then(response => response.json())
.then(data => {

suggestions.innerHTML = "";

data.forEach(word => {

let div = document.createElement("div");
div.classList.add("item");
div.textContent = word;

div.onclick = function(){
searchBox.value = word;
suggestions.innerHTML = "";
}

suggestions.appendChild(div);

});

});

});

</script>

</body>
</html>

