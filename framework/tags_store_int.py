#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tags_store.py

"""
from typing import List, Set, Iterator, Tuple, Dict
import gzip

__author__ = "We-Te Chen"


def fopen(file_name: str):
    if file_name.endswith('.gz'):
        return gzip.open(file_name, 'rt')
    return open(file_name, 'r')


class TagMaster(object):
    tag_group_id_idx = 0
    tag_id_idx = 2
    tag_id_name_idx = 3
    tag_is_leaf_idx = 10

    def __init__(self, tag_master_file: str):
        self.tag_id_name_map = {}
        self.tag_id_group_id_map = {}
        self.group_id_tag_id_map = {}
        with fopen(tag_master_file) as fhd:
            for line in fhd:
                terms = line.strip().split("\t")

                if terms[TagMaster.tag_is_leaf_idx].lower() == "true":
                    tag_group_id = int(terms[TagMaster.tag_group_id_idx])
                    tag_id = int(terms[TagMaster.tag_id_idx])
                    tag_id_name = terms[self.tag_id_name_idx]

                    self.tag_id_name_map[tag_id] = tag_id_name

                    self.tag_id_group_id_map[tag_id] = tag_group_id

                    if tag_group_id not in self.group_id_tag_id_map:
                        self.group_id_tag_id_map[tag_group_id] = set()
                    self.group_id_tag_id_map[tag_group_id].add(tag_id)


class GenreTaxonomy(object):
    class GenreNode(object):
        def __init__(self, genre, parent_node):
            self.genre = genre
            self.parent = parent_node
            self.child = []

        def add_child(self, child_node):
            self.child.append(child_node)

        def get_child(self, recursive=False):
            if recursive:
                return [self.genre] + [d_genre for c_node in self.child for d_genre in c_node.get_child(recursive)]
            else:
                return [self.genre] + [c_node.genre for c_node in self.child]

    def __init__(self, genre_master_file: str):
        self.root = GenreTaxonomy.GenreNode(0, None)
        current_node = self.root
        self.genre_map = {}
        with fopen(genre_master_file) as fhd:
            fhd.readline()
            for line in fhd:
                (genre_id_str, genre_path_str, _, _) = line.split('\t')
                genre_id = int(genre_id_str)

                last_idx = genre_path_str.rfind('/')
                last2_idx = genre_path_str.rfind('/', 0, last_idx)
                parent_genre_id = int(genre_path_str[last2_idx+1:last_idx])
                while current_node.genre != parent_genre_id:
                    current_node = current_node.parent

                new_node = GenreTaxonomy.GenreNode(genre_id, current_node)
                self.genre_map[genre_id] = new_node
                current_node.add_child(new_node)
                current_node = new_node

    def is_leaf_genre(self, genre_id: int) -> bool:
        genre_node = self.genre_map[genre_id]
        return not genre_node.child

    def get_all_child_genres(self, genre_id: int) -> List[str]:
        genre_node = self.genre_map[genre_id]

        return genre_node.get_child(recursive=True)

    def trace_root_genre(self, genre_id: int) -> Iterator[int]:
        genre_node = self.genre_map[genre_id]
        while genre_node.parent:
            yield genre_node.genre
            genre_node = genre_node.parent

    def get_child_leaf_genres(self, genre_id: int) -> Iterator[int]:
        genre_node = self.genre_map[genre_id]
        if genre_node.child:
            for c_node in genre_node.child:
                for l_genre in self.get_child_leaf_genres(c_node.genre):
                    yield l_genre
        else:
            yield genre_node.genre


class TagStore(object):
    """ A collection of Ichiba tags collection and method

    """
    @staticmethod
    def get_tag_genre_dicts(tag_master_file: str, genre_master_file: str, genre_tag_file: str,
                            genre_gms_file: str = None) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        """

        :param tag_master_file:  Tag Master File
        :param genre_master_file: Genre File
        :param genre_tag_file: Genre - Tag ID mapping file
        :param genre_gms_file: Genre GMS file
        :return: Tuple of (1) Genre to Tag Group ID set, and (2) Tag Group ID to Genre set
        """
        tag_master = TagMaster(tag_master_file)
        genre_taxonomy = GenreTaxonomy(genre_master_file)

        genre_tag_group_map = {}
        tag_group_genre_map = {}
        genre_gms_list = []

        genre2tag_group = {}

        with fopen(genre_tag_file) as fhd:
            fhd.readline()
            for line in fhd:
                (genre_str, tag_str) = line.strip().split('\t')
                genre = int(genre_str)
                tag_list = [int(tag) for tag in tag_str.split(',')]

                # retrieve tag_group_set
                tag_group_set = set()
                for tag_id in tag_list:
                    try:
                        tag_group = tag_master.tag_id_group_id_map[tag_id]
                    except KeyError:
                        pass
                    else:
                        tag_group_set.add(tag_group)

                genre_tag_group_map[genre] = tag_group_set

                if genre in genre_taxonomy.genre_map:
                    for tag_group in tag_group_set:
                        if tag_group not in tag_group_genre_map:
                            tag_group_genre_map[tag_group] = {genre}
                        else:
                            tag_group_genre_map[tag_group].add(genre)

        def link_tag_group(g_node: GenreTaxonomy.GenreNode, parent_tag_group_set=None):
            if not parent_tag_group_set:
                parent_tag_group_set = set()

            if g_node.genre in genre_tag_group_map:
                _tag_group_set = parent_tag_group_set.union(genre_tag_group_map[g_node.genre])
            else:
                _tag_group_set = parent_tag_group_set

            if g_node.child:
                for c_node in g_node.child:
                    link_tag_group(c_node, _tag_group_set)
            else:
                genre2tag_group[g_node.genre] = _tag_group_set

        link_tag_group(genre_taxonomy.root)

        for (tag_group, genre_set) in tag_group_genre_map.items():
            tag_group_genre_map[tag_group] =\
                set(c_genre for genre in genre_set for c_genre in genre_taxonomy.get_child_leaf_genres(genre))

        if genre_gms_file:
            with fopen(genre_gms_file) as fhd:
                for line in fhd:
                    (genre_str, revenue) = line.strip().split('\t')
                    genre_gms_list.append((genre_str, int(revenue)))

        return genre2tag_group, tag_group_genre_map