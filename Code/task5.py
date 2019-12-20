import os
import numpy as np
import h5py
import json
import pickle
import base64
import webbrowser

from sklearn.decomposition import PCA

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
# print("Current_dir: ",CURRENT_DIR)
root_path = CURRENT_DIR+"/../Metadata/"
hdf5_file = root_path + "feature_vectors_full_data.hdf5"


def euclidean_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    dist = np.sum((x-y)**2)
    return dist


def euclidean_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    dist = np.sum((x-y)**2)
    return 1/(dist+1.0)


class LSHIndex:
    def __init__(self, load_sift=False):
        with h5py.File(hdf5_file, 'r') as hf:
            # data_hog = hf['hog_features'][:]
            self.image_ids = hf['image_ids'][:]
            # data_cm = hf["cm_features"][:]
            # print(list(hf.keys()))
    
        with h5py.File(hdf5_file, 'r') as hf:
            if 'hog_pca_256' not in hf.keys():
                print("hog_pca not in file, creating")
                data_hog = hf['hog_features'][:]
                pca_hog = PCA(n_components=256)
                self.data_hog_ = pca_hog.fit_transform(data_hog)
                hf.create_dataset('hog_pca_256', data=self.data_hog_)
                print("hog_pca_256 created and stored in file")
            else:
                self.data_hog_ = hf['hog_pca_256'][:]
                # print("hog_pca_256 loaded from file")
            if load_sift:
                self.data_sift = hf["sift_features"][:]
                self.data_sift_kp_starts = hf["sift_kp_starts"][:]

        with h5py.File(hdf5_file, 'r') as hf:
            if 'cm_pca_256' not in hf.keys():
                print("cm_pca not in file, creating")
                data_cm = hf["cm_features"][:]
                pca_cm = PCA(n_components=256)
                self.data_cm_ = pca_cm.fit_transform(data_cm)
                hf.create_dataset('cm_pca_256', data=self.data_cm_)
                print("cm_pca_256 created and stored in file")
            else:
                self.data_cm_ = hf['cm_pca_256'][:]
                # print("cm_pca_256 loaded from file")

        self.image_id_map = dict()  # a mape from image name to index in hdf5 file
        for i in range(len(self.image_ids)):
            a = self.image_ids[i].decode('UTF-8')
            self.image_id_map[a] = i
        self.index_structure = None
        self.index_in_memory = False
        self.index_file = CURRENT_DIR+"/../Outputs/Task_5/hands_index.pkl"  # "./hands_index.pkl"
        self.output_file = CURRENT_DIR+"/../Outputs/Task_5/task5_output.pkl"  # "task5_output.pkl"
        self.hands_path = CURRENT_DIR+"/../../Hands"  # "./../../Hands"
        self.html_file = CURRENT_DIR+"/../Outputs/Task_5/task5_viz.html"
        self.t_input = -1
        self.ranked_element_list = []  # initially the ranked list is null
        
    def store_index(self):
        # Store index to File
        
        with open(self.index_file, "wb") as fp:
            pickle.dump(self.index_structure, fp)
            
    def create_index(self):
        data_ = self.data_hog_
        org_data = self.data_hog_
        k = int(input("Enter the number of hashes per layer\n"))
        L = int(input("Enter the number of Layers\n"))
        w = k+9  # hog: 17, cm: 600
        d = data_.shape[1]
        np.random.seed(83)
        G = np.random.normal(size=(L, k, d))  # the random vectors for w
        B = np.random.uniform(low=0, high=w, size=(L, k))  # the random scalars from uniform distribution

        hash_tables = {}
        for layer_i in range(L):    
            f1 = G[layer_i]  # v's for first hash Family, containing k vectors of size d.
            b1 = B[layer_i]  # b's for first hash Family, containing k scalar values
            # print(f1.shape)
            print("creating layer: ", layer_i)
            # hash_tables[layer_i]
            cluster = dict() # cluster will contain points whose all hash values are the same.
            img_cluster_map = dict()
            for i in range(len(data_)):  # pick each data point
                d = data_[i]
                g = []  # g will be the concatenated hash value for the data point
                for j in range(len(f1)):  # for each hash function
                    v = f1[j]
                    b = b1[j]
                    h = np.floor( (np.dot(d, v) + b)/w )
                    g.append(int(h))
                g_ = json.dumps(g)
                if g_ not in cluster:
                    cluster[g_] = set()
                img_cluster_map[i] = g
                cluster[g_].add(i)
            hash_tables[layer_i] = cluster
            
        index_structure = dict()
        index_structure['hash_tables'] = hash_tables
        index_structure['k'] = k
        index_structure['L'] = L
        index_structure['w'] = w
        index_structure['G'] = G
        index_structure['B'] = B
        
        self.index_structure = index_structure
        self.index_in_memory = True
        
        self.store_index()
            
    def retrieve_index(self):
        if self.index_in_memory:
            return 1
        else:  # Needs to retrieve index from disk
            try:
                with open(self.index_file, "rb") as fp:
                    self.index_structure = pickle.load(fp)
                    self.index_in_memory = True
                    return 1
            except FileNotFoundError:
                print("Index not on disk. Please create an index first")
                return 0
            
    def query_image_task(self):
        query_image_name = str(input("Enter image id (just the number)\n"))
        t_input = int(input("Enter number of similar images to be shown: "))
        len_name = len(query_image_name)
        query_image_name = '0'*(7-len_name)+query_image_name
        self.t_input = t_input
        
        self.query_image(query_image_name, inside_call=True)
        query_index = self.image_id_map[query_image_name]
        self.visualize_similar_results(query_index, t_input, self.ranked_element_list)

    def query_image(self, image_id, inside_call=False):  # image_id(string): "0000674"
        
        retrieve_index_result = self.retrieve_index()
        if retrieve_index_result == 0:
            self.create_index()
        
        query = self.image_id_map[image_id]
        
        hash_tables = self.index_structure['hash_tables']
        k = self.index_structure['k']
        L = self.index_structure['L'] 
        w = self.index_structure['w']
        G = self.index_structure['G']
        B = self.index_structure['B']
        
        data_ = self.data_hog_
        
        hash_values = {}

        for layer_i in range(L):    
            f1 = G[layer_i]  # v's for first hash Family, containing k vecctors of size d.
            b1 = B[layer_i]  # b's for first hash Family, containing k scalar values
            d = data_[query]
            g = []  # g will be the concatenated hash value for the data point
            for j in range(len(f1)):  # for each hash function
                v = f1[j]
                b = b1[j]
                h = np.floor( (np.dot(d, v) + b)/w )
                g.append(int(h))
            g_ = json.dumps(g)
            # print(g_)
            hash_values[layer_i] = g_
            
        solution = set()  # ID of matched results
        sol_element = set() # index of matched elements
        total_count = 0
        for k, v in hash_values.items():  # k is the Layer number, v is the HashValue for Layer L
            table = hash_tables[k]  # retrieve hash table of whole 11k data
            if v in table:
                # print(table[v])
                for element in table[v]:
                    total_count += 1
                    sol_element.add(element)
                    solution.add(self.image_ids[element])
        if inside_call:
            print("total_count = ", total_count)
            print("unique_count = ", len(sol_element))
        
        self.get_ranked_images(query, sol_element, inside_call=inside_call)
        
        return self.ranked_element_list

    def get_ranked_images(self, query_index, similar_element_list, inside_call=False):
        similarity_list = []
        query_v = self.data_hog_[query_index]
        for i in similar_element_list:
            img_u = self.data_hog_[i]
            eu_sim = euclidean_similarity(query_v, img_u)
            similarity_list.append((self.image_ids[i].decode('UTF-8'), i, eu_sim))
            
        self.ranked_element_list = sorted(similarity_list, key=lambda x: x[2], reverse=True)

        if inside_call:
            if self.t_input > 0:
                with open(self.output_file, "wb") as fp:
                    pickle.dump(self.ranked_element_list[:self.t_input], fp)
            else:
                with open(self.output_file, "wb") as fp:
                    pickle.dump(self.ranked_element_list, fp)

    def visualize_similar_results(self, query_index, t_input, ranked_element_list):
        fig_tags = []
        counter = 0
        
        query_image_name = self.image_ids[query_index].decode('UTF-8')
        print("query_image_name: ", query_image_name)
        
        for u, v, w in ranked_element_list[:t_input]:
            img_id = u
            counter += 1
            image_name = 'Hand_{0}.jpg'.format(img_id)
            data_uri = base64.b64encode(open(self.hands_path+os.sep+image_name, 'rb').read()).decode('utf-8')

            if counter == 1:
                figure_caption = "Retrieved Image: {1},<br/> Score: {2}"\
                        .format(query_image_name,
                                image_name,w)
            else:
                figure_caption = "Image: {0}, Score: {1}".format(image_name, w)
            caption_tag = '<figcaption>{0}</figcaption>'.format(figure_caption)
            img_tag = '<img src="data:image/png;base64,{0}" style="width:400px;height:300px;">'.format(data_uri)
            figure_tag = "<figure>"+caption_tag+img_tag+"</figure>"

            fig_tags.append(figure_tag)

        f = open(self.html_file, 'w')

        message = """
        <html>
        <head>
        <style>
            figure {
                width: 19% !important;
                float: left !important;
            }
            
            figure img {
                max-width: 100%;
                height: auto !important;
                width: 100% !important;
            }
            
            figcaption {
                padding: 2% 0 !important;
            }
        </style>
        </head>
        <body>
        """

        message += "<br/><div>Query image ID:{0}</div>".format(query_image_name)

        for figs in fig_tags:
            message = message + figs

        message = message +\
            """
            </body>
            </html>
            """

        f.write(message)
        f.close()

        # webbrowser.open_new_tab('images.html')
        webbrowser.open('file://' + os.path.realpath(self.html_file))

    def visualize_similar_results_v2(self, query_image_name, ranked_image_ids, scores, t_input):
        fig_tags = []
        counter = 0

        print("query_image_name: ", query_image_name)

        for ranked_image_id, score in zip(ranked_image_ids[:t_input], scores):
            counter += 1
            image_name = 'Hand_{0}.jpg'.format(ranked_image_id)
            data_uri = base64.b64encode(open(self.hands_path + os.sep + image_name, 'rb').read()).decode('utf-8')

            if counter == 1:
                figure_caption = "Retrieved Image: {1},<br/> Score: {2}" \
                    .format(query_image_name,
                            image_name, score)
            else:
                figure_caption = "Image: {0}, Score: {1}".format(image_name, score)
            caption_tag = '<figcaption>{0}</figcaption>'.format(figure_caption)
            img_tag = '<img src="data:image/png;base64,{0}" style="width:400px;height:300px;">'.format(data_uri)
            figure_tag = "<figure>" + caption_tag + img_tag + "</figure>"

            fig_tags.append(figure_tag)

        f = open(self.html_file, 'w')

        message = """
        <html>
        <head>
        <style>
            figure {
                width: 19% !important;
                float: left !important;
            }

            figure img {
                max-width: 100%;
                height: auto !important;
                width: 100% !important;
            }

            figcaption {
                padding: 2% 0 !important;
            }
        </style>
        </head>
        <body>
        """

        message += "<br/><div>Query image ID:{0}</div>".format(query_image_name)

        for figs in fig_tags:
            message = message + figs

        message = message + \
                  """
                  </body>
                  </html>
                  """

        f.write(message)
        f.close()

        # webbrowser.open_new_tab('images.html')
        webbrowser.open('file://' + os.path.realpath(self.html_file))
        

if __name__ == "__main__":
    index1 = LSHIndex()
    
    while True:
        print("Enter 1 to create index\nEnter 2 to query from already created index")
        print("Enter 3 to exit")
        choice = int(input("Your choice: "))
        if choice == 1:
            index1.create_index()
        elif choice == 2:
            index1.query_image_task()
        elif choice == 3:
            print("Exited.")
            break
        else:
            print("Wrong Input!")
