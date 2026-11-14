// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

namespace
{
struct PointerProperty
{
	const char* tokens[6]; // suffix tokens after the root object, terminated by NULL
	size_t components;
};

// suffixes of JSON pointers that gltfpack can preserve when animating materials via KHR_animation_pointer
// the root token (e.g. /materials/0) is matched separately; the table below lists every supported property path
const PointerProperty kMaterialProperties[] = {
	{{"pbrMetallicRoughness", "baseColorFactor", NULL}, 4},
	{{"pbrMetallicRoughness", "metallicFactor", NULL}, 1},
	{{"pbrMetallicRoughness", "roughnessFactor", NULL}, 1},
	{{"emissiveFactor", NULL}, 3},
	{{"alphaCutoff", NULL}, 1},
	{{"normalTexture", "scale", NULL}, 1},
	{{"occlusionTexture", "strength", NULL}, 1},
	{{"pbrMetallicRoughness", "baseColorTexture", "extensions", "KHR_texture_transform", "offset", NULL}, 2},
	{{"pbrMetallicRoughness", "baseColorTexture", "extensions", "KHR_texture_transform", "rotation", NULL}, 1},
	{{"pbrMetallicRoughness", "baseColorTexture", "extensions", "KHR_texture_transform", "scale", NULL}, 2},
	{{"pbrMetallicRoughness", "metallicRoughnessTexture", "extensions", "KHR_texture_transform", "offset", NULL}, 2},
	{{"pbrMetallicRoughness", "metallicRoughnessTexture", "extensions", "KHR_texture_transform", "rotation", NULL}, 1},
	{{"pbrMetallicRoughness", "metallicRoughnessTexture", "extensions", "KHR_texture_transform", "scale", NULL}, 2},
	{{"normalTexture", "extensions", "KHR_texture_transform", "offset", NULL}, 2},
	{{"normalTexture", "extensions", "KHR_texture_transform", "rotation", NULL}, 1},
	{{"normalTexture", "extensions", "KHR_texture_transform", "scale", NULL}, 2},
	{{"occlusionTexture", "extensions", "KHR_texture_transform", "offset", NULL}, 2},
	{{"occlusionTexture", "extensions", "KHR_texture_transform", "rotation", NULL}, 1},
	{{"occlusionTexture", "extensions", "KHR_texture_transform", "scale", NULL}, 2},
	{{"emissiveTexture", "extensions", "KHR_texture_transform", "offset", NULL}, 2},
	{{"emissiveTexture", "extensions", "KHR_texture_transform", "rotation", NULL}, 1},
	{{"emissiveTexture", "extensions", "KHR_texture_transform", "scale", NULL}, 2},
};

const PointerProperty kLightProperties[] = {
	{{"color", NULL}, 3},
	{{"intensity", NULL}, 1},
	{{"range", NULL}, 1},
};

std::string unescapeToken(const char* begin, const char* end)
{
	std::string result;
	result.reserve(size_t(end - begin));

	for (const char* it = begin; it != end; ++it)
	{
		if (*it == '~' && it + 1 != end)
		{
			if (it[1] == '0')
			{
				result.push_back('~');
				++it;
				continue;
			}
			else if (it[1] == '1')
			{
				result.push_back('/');
				++it;
				continue;
			}
		}

		result.push_back(*it);
	}

	return result;
}

std::vector<std::string> splitPointer(const std::string& uri)
{
	std::vector<std::string> tokens;

	size_t pos = 0;
	while (pos < uri.size())
	{
		if (uri[pos] != '/')
			break;

		size_t next = uri.find('/', pos + 1);
		size_t end = (next == std::string::npos) ? uri.size() : next;

		tokens.push_back(unescapeToken(uri.c_str() + pos + 1, uri.c_str() + end));
		pos = end;
	}

	return tokens;
}

bool parseIndex(const std::string& token, size_t& index)
{
	if (token.empty())
		return false;

	size_t value = 0;

	for (size_t i = 0; i < token.size(); ++i)
	{
		if (token[i] < '0' || token[i] > '9')
			return false;

		value = value * 10 + size_t(token[i] - '0');

		if (value > (1ull << 31))
			return false;
	}

	index = value;
	return true;
}

bool matchProperty(const std::vector<std::string>& tokens, size_t offset, const PointerProperty& property)
{
	size_t count = 0;
	while (property.tokens[count])
		++count;

	if (tokens.size() - offset != count)
		return false;

	for (size_t i = 0; i < count; ++i)
		if (tokens[offset + i] != property.tokens[i])
			return false;

	return true;
}

const PointerProperty* findProperty(const PointerProperty* properties, size_t count, const std::vector<std::string>& tokens, size_t offset)
{
	for (size_t i = 0; i < count; ++i)
		if (matchProperty(tokens, offset, properties[i]))
			return &properties[i];

	return NULL;
}

// extracts the "uri" string of a KHR_animation_pointer extension object
// the extension data is guaranteed to be a JSON object ({"uri":"..."}), so a small hand-rolled parser keeps this file independent of cgltf's jsmn
const char* parseExtensionUri(const char* json, std::string& uri)
{
	const char* it = json;

	while (*it && isspace(static_cast<unsigned char>(*it)))
		++it;

	if (*it++ != '{')
		return "invalid KHR_animation_pointer extension";

	while (*it && isspace(static_cast<unsigned char>(*it)))
		++it;

	if (*it == '}')
		return "KHR_animation_pointer extension is missing uri";

	while (*it)
	{
		while (*it && isspace(static_cast<unsigned char>(*it)))
			++it;

		if (*it != '"')
			return "invalid KHR_animation_pointer extension";
		++it;

		std::string key;
		while (*it && *it != '"')
		{
			if (*it == '\\' && it[1])
				++it;
			key.push_back(*it++);
		}
		if (*it != '"')
			return "invalid KHR_animation_pointer extension";
		++it;

		while (*it && isspace(static_cast<unsigned char>(*it)))
			++it;
		if (*it++ != ':')
			return "invalid KHR_animation_pointer extension";
		while (*it && isspace(static_cast<unsigned char>(*it)))
			++it;

		if (key == "uri")
		{
			if (*it != '"')
				return "KHR_animation_pointer uri must be a string";
			++it;

			uri.clear();
			while (*it && *it != '"')
			{
				if (*it == '\\' && it[1])
					++it; // preserve escape sequences verbatim; they are re-emitted through appendJson unchanged
				uri.push_back(*it++);
			}
			if (*it != '"')
				return "invalid KHR_animation_pointer extension";

			return NULL;
		}

		// skip unknown value (string, object, array, number, true/false/null)
		if (*it == '"')
		{
			++it;
			while (*it && *it != '"')
			{
				if (*it == '\\' && it[1])
					++it;
				++it;
			}
			if (*it != '"')
				return "invalid KHR_animation_pointer extension";
			++it;
		}
		else if (*it == '{' || *it == '[')
		{
			char open = *it;
			char close = open == '{' ? '}' : ']';
			int depth = 0;
			bool in_string = false;

			do
			{
				if (*it == '"')
					in_string = !in_string;
				else if (!in_string && *it == open)
					++depth;
				else if (!in_string && *it == close)
					--depth;
				++it;
			} while (*it && depth > 0);

			if (depth != 0)
				return "invalid KHR_animation_pointer extension";
		}
		else
		{
			while (*it && *it != ',' && *it != '}')
				++it;
		}

		while (*it && isspace(static_cast<unsigned char>(*it)))
			++it;
		if (*it == ',')
		{
			++it;
			continue;
		}
		if (*it == '}')
			break;

		return "invalid KHR_animation_pointer extension";
	}

	return "KHR_animation_pointer extension is missing uri";
}
} // namespace

const char* parseAnimationPointer(const cgltf_animation_channel& channel, cgltf_data* data, size_t accessor_components, PointerTarget& target, std::string& pointer)
{
	target = PointerTarget();
	pointer.clear();

	const cgltf_extension* extension = NULL;

	for (size_t i = 0; i < channel.extensions_count; ++i)
	{
		if (strcmp(channel.extensions[i].name, "KHR_animation_pointer") == 0)
		{
			extension = &channel.extensions[i];
			break;
		}
	}

	if (!extension)
		return NULL;

	const char* error = parseExtensionUri(extension->data, pointer);
	if (error)
		return error;

	if (pointer.empty() || pointer[0] != '/')
		return "KHR_animation_pointer uri must be an absolute JSON pointer";

	std::vector<std::string> tokens = splitPointer(pointer);
	if (tokens.size() < 3)
		return "KHR_animation_pointer uri does not reference an animatable property";

	size_t root_index = 0;

	const PointerProperty* property = NULL;

	if (tokens[0] == "materials")
	{
		if (!parseIndex(tokens[1], root_index))
			return "KHR_animation_pointer uri references an invalid object index";

		if (root_index >= data->materials_count)
			return "KHR_animation_pointer uri references a material that does not exist";

		property = findProperty(kMaterialProperties, sizeof(kMaterialProperties) / sizeof(kMaterialProperties[0]), tokens, 2);

		if (!property)
			return "KHR_animation_pointer uri references an unsupported material property";

		target.kind = PointerTarget_Material;
		target.index = root_index;
	}
	else if (tokens[0] == "extensions" && tokens.size() >= 5 && tokens[1] == "KHR_lights_punctual" && tokens[2] == "lights")
	{
		if (!parseIndex(tokens[3], root_index))
			return "KHR_animation_pointer uri references an invalid object index";

		if (root_index >= data->lights_count)
			return "KHR_animation_pointer uri references a light that does not exist";

		property = findProperty(kLightProperties, sizeof(kLightProperties) / sizeof(kLightProperties[0]), tokens, 4);

		if (!property)
			return "KHR_animation_pointer uri references an unsupported light property";

		target.kind = PointerTarget_Light;
		target.index = root_index;
	}
	else
	{
		return "KHR_animation_pointer uri references an unsupported object; only materials and KHR_lights_punctual lights are supported";
	}

	if (accessor_components != property->components)
		return "KHR_animation_pointer channel output component count does not match the target property";

	target.components = property->components;
	target.suffix.assign(tokens.begin() + (target.kind == PointerTarget_Material ? 2 : 4), tokens.end());

	return NULL;
}

const char* validateAnimationPointers(const std::vector<Animation>& animations, const std::vector<MaterialInfo>& materials)
{
	for (size_t i = 0; i < animations.size(); ++i)
	{
		const Animation& animation = animations[i];

		for (size_t j = 0; j < animation.tracks.size(); ++j)
		{
			const Track& track = animation.tracks[j];

			if (track.pointer.empty())
				continue;

			if (track.target.kind == PointerTarget_Material && !materials[track.target.index].keep)
			{
				fprintf(stderr, "Error: KHR_animation_pointer target material %zu was removed during optimization\n", track.target.index);
				return "KHR_animation_pointer target was removed during optimization";
			}
		}
	}

	return NULL;
}

void remapAnimationPointerMaterials(std::vector<Animation>& animations, cgltf_material* const* material_remap, cgltf_data* data)
{
	for (size_t i = 0; i < animations.size(); ++i)
	{
		for (size_t j = 0; j < animations[i].tracks.size(); ++j)
		{
			Track& track = animations[i].tracks[j];

			if (!track.pointer.empty() && track.target.kind == PointerTarget_Material)
			{
				const cgltf_material* canonical = material_remap[track.target.index];
				track.target.index = size_t(canonical - data->materials);
			}
		}
	}
}

static void appendEscapedToken(std::string& uri, const std::string& token)
{
	uri.push_back('/');

	for (size_t i = 0; i < token.size(); ++i)
	{
		switch (token[i])
		{
		case '~':
			uri += "~0";
			break;
		case '/':
			uri += "~1";
			break;
		default:
			uri.push_back(token[i]);
			break;
		}
	}
}

std::string rewriteAnimationPointer(const PointerTarget& target, const std::vector<MaterialInfo>& materials)
{
	std::string uri;

	if (target.kind == PointerTarget_Material)
	{
		uri += "/materials/";
		append(uri, size_t(materials[target.index].remap));
	}
	else
	{
		assert(target.kind == PointerTarget_Light);

		uri += "/extensions/KHR_lights_punctual/lights/";
		append(uri, target.index);
	}

	for (size_t i = 0; i < target.suffix.size(); ++i)
		appendEscapedToken(uri, target.suffix[i]);

	return uri;
}
