from google_manager import detect_labels
from bing_search import bing_annotate
from annotator import rename_objXML, rename_objXML_1, rename_objXML_2
import os
import argparse
import shutil

def match_labels(bing_labels, google_labels,file):
    """Match labels will compare the key values of the two dictionaries and compute the mean value of the given propabilities

    Args:
        bing_labels (dictionary): key value is the category, value is the propability
        google_labels (_type_): key value is the category, value is the propability

    Returns:
        dictionary: key value is the matched category, value is the propability
    """
    # Create a dictionary to store matched labels with their mean scores
    matched_labels = {}
    for key in bing_labels[file]:
        if key in google_labels.keys():
            #print(f"{key.lower()} (bing score: {bing_labels[file][key]}) (google score: {new_google_labels[key] })")
            # Calculate the mean score for the matched label
            matched_score = [bing_labels[file][key] ]
            matched_score.append(google_labels[key])
            mean_score = sum(matched_score) / len(matched_score)
            # Add the matched label and its mean score to the dictionary
            matched_labels[key] = mean_score
    return matched_labels


def get_web_labels_both(file):

    google_labels = detect_labels(file)
    bing_labels = bing_annotate(file)

    new_google_labels = {}
    for label in google_labels:
        new_google_labels[label.description.lower()] = label.score
    
    return new_google_labels, bing_labels


def get_args_parser():
    parser = argparse.ArgumentParser('Web based categorisation', add_help=False)
    parser.add_argument('--engine', default='google', type=str, choices=('google', 'bing', 'gb'), help='choose an search engine')
    return parser


def annotate_unknowns(args , unknown_dir, annotate_dir):
    for unknown in os.listdir(unknown_dir):
        #print(unknown.split('_')[-2])
        threshold = 0.85
        best_score = 0.0
        best_label = ''

        print(f"\nSearching in web for picture:{unknown} ")
        file = os.path.join(unknown_dir, unknown)
        i = int(file.split('.')[0].split('_')[-1])

        labels = {}
        if args.engine == 'google' or args.engine == 'bing':
            if args.engine == 'google':
                google_labels =detect_labels(file)
                for label in google_labels:
                    #print(f'{label.description} (score: {label.score})')
                    labels[label.description.lower()] = label.score

            else:
                labels = bing_annotate(file)[file]
            
            annotate_file = os.path.splitext(os.path.join(annotate_dir, unknown.split('_')[-2]))[0] + ".xml" #unknown
        
            for key in labels:
                #print(f"Label:{key} {new_google_labels[key]}")
                if best_score < labels[key]:
                    best_score = labels[key]
                    best_label = key.lower()
            if best_score > threshold:
                print(f"Reannotating Best found label {best_label} (score: {best_score})")
                rename_objXML_2(annotate_file , i, best_label.lower())
            
        elif args.engine == 'gb':
            print(f"\n____ Fusing Google and Bing____")

            new_google_labels, bing_labels = get_web_labels_both(file)
            # Create a dictionary to store matched labels with their mean scores
            matched_labels = match_labels(bing_labels, new_google_labels, file)

            # Print the matched labels and their mean scores
            if len(matched_labels)==0:
                print(f"Nothing matched!")
                continue
            else:
                best_score = 0 
                best_label = ''
                print(f"Matched labels: {matched_labels.items()}")
                for label, mean_score in matched_labels.items():
                    print(f"{label} (mean score: {mean_score})")
                    if mean_score > threshold:
                        if mean_score > best_score:
                            best_score = mean_score
                            best_label = label
                if best_score>threshold:
                    #print(f"\nUnknown {i} will be reannotated to {best_label} (score: {best_score})\n")
                    annotate_file = os.path.splitext(os.path.join(annotate_dir, unknown.split('_')[-2]))[0] + ".xml" #unknown
                    rename_objXML_2(annotate_file , i, best_label.lower())
        else:
            print("You must choose an engine in order to choose the labels.")


def main(args):
    unknown_folder =os.path.join(os.getcwd(), 'output', 'unknown')
    annotate_dir = os.path.join(os.getcwd(), 'output', 'Annotations')

    for folder in os.listdir(unknown_folder):
        unknown_dir = os.path.join(unknown_folder, folder)
        annotate_unknowns(args, unknown_dir, annotate_dir)
    for folder in os.listdir(unknown_folder):
        unknown_dir = os.path.join(unknown_folder, folder)
        shutil.rmtree(unknown_dir, ignore_errors=True)


               

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Retrieving the unknown instances using search engine', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)