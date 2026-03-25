import React, { useMemo, useState } from "react";
import { StatusBar } from "expo-status-bar";
import * as ImagePicker from "expo-image-picker";
import {
    ActivityIndicator,
    Alert,
    Image,
    KeyboardAvoidingView,
    Platform,
    Pressable,
    SafeAreaView,
    ScrollView,
    StyleSheet,
    Text,
    TextInput,
    View,
} from "react-native";

const DEFAULT_API_BASE_URL = "";

function trimTrailingSlash(value) {
    return value.replace(/\/+$/, "");
}

function guessMimeType(fileName) {
    const lower = (fileName || "").toLowerCase();
    if (lower.endsWith(".png")) {
        return "image/png";
    }
    if (lower.endsWith(".webp")) {
        return "image/webp";
    }
    return "image/jpeg";
}

export default function App() {
    const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
    const [selectedAsset, setSelectedAsset] = useState(null);
    const [result, setResult] = useState(null);
    const [errorMessage, setErrorMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const normalizedApiBaseUrl = useMemo(
        () => trimTrailingSlash(apiBaseUrl.trim()),
        [apiBaseUrl]
    );

    async function pickImage() {
        try {
            const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (!permission.granted) {
                Alert.alert(
                    "Permission required",
                    "Allow photo library access to choose an image for analysis."
                );
                return;
            }

            const pickerResult = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ["images"],
                allowsEditing: false,
                quality: 1,
            });

            if (pickerResult.canceled || !pickerResult.assets?.length) {
                return;
            }

            setSelectedAsset(pickerResult.assets[0]);
            setResult(null);
            setErrorMessage("");
        } catch (error) {
            setErrorMessage(error.message || "Unable to open image picker.");
        }
    }

    async function analyzeImage() {
        if (!selectedAsset) {
            Alert.alert("Choose an image", "Select an image before running analysis.");
            return;
        }

        if (!normalizedApiBaseUrl) {
            Alert.alert(
                "Set API URL",
                "Enter the Flask backend URL, for example your deployed Render URL."
            );
            return;
        }

        setIsLoading(true);
        setErrorMessage("");
        setResult(null);

        try {
            const formData = new FormData();
            const fileName = selectedAsset.fileName || "mobile-upload.jpg";
            formData.append("image", {
                uri: selectedAsset.uri,
                name: fileName,
                type: selectedAsset.mimeType || guessMimeType(fileName),
            });

            const response = await fetch(`${normalizedApiBaseUrl}/api/detect_ai`, {
                method: "POST",
                body: formData,
                headers: {
                    Accept: "application/json",
                },
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Analysis failed.");
            }

            setResult(data);
        } catch (error) {
            setErrorMessage(error.message || "Analysis failed.");
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <SafeAreaView style={styles.safeArea}>
            <StatusBar style="light" />
            <KeyboardAvoidingView
                style={styles.flex}
                behavior={Platform.OS === "ios" ? "padding" : undefined}
            >
                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    keyboardShouldPersistTaps="handled"
                >
                    <View style={styles.hero}>
                        <Text style={styles.kicker}>AAD Mobile</Text>
                        <Text style={styles.title}>React Native showcase client</Text>
                        <Text style={styles.subtitle}>
                            Pick one image, call the existing Flask API, and show the AI
                            detection verdict on mobile.
                        </Text>
                    </View>

                    <View style={styles.card}>
                        <Text style={styles.label}>Backend URL</Text>
                        <TextInput
                            style={styles.input}
                            placeholder="https://your-render-url.onrender.com"
                            placeholderTextColor="#7b8b91"
                            autoCapitalize="none"
                            autoCorrect={false}
                            keyboardType="url"
                            value={apiBaseUrl}
                            onChangeText={setApiBaseUrl}
                        />
                        <Text style={styles.helpText}>
                            Use your deployed Flask URL or a LAN IP when testing locally.
                        </Text>
                    </View>

                    <View style={styles.card}>
                        <Text style={styles.label}>Selected image</Text>
                        {selectedAsset ? (
                            <View style={styles.previewWrap}>
                                <Image source={{ uri: selectedAsset.uri }} style={styles.preview} />
                                <Text style={styles.previewMeta}>
                                    {selectedAsset.fileName || "mobile-upload.jpg"}
                                </Text>
                            </View>
                        ) : (
                            <Text style={styles.emptyState}>No image selected yet.</Text>
                        )}

                        <View style={styles.buttonRow}>
                            <Pressable style={styles.secondaryButton} onPress={pickImage}>
                                <Text style={styles.secondaryButtonText}>Choose Image</Text>
                            </Pressable>
                            <Pressable
                                style={[
                                    styles.primaryButton,
                                    (isLoading || !selectedAsset) && styles.buttonDisabled,
                                ]}
                                onPress={analyzeImage}
                                disabled={isLoading || !selectedAsset}
                            >
                                <Text style={styles.primaryButtonText}>
                                    {isLoading ? "Analyzing..." : "Analyze"}
                                </Text>
                            </Pressable>
                        </View>
                    </View>

                    {isLoading ? (
                        <View style={styles.card}>
                            <View style={styles.loadingRow}>
                                <ActivityIndicator size="small" color="#0b8f8a" />
                                <Text style={styles.loadingText}>
                                    Uploading image and waiting for detector output...
                                </Text>
                            </View>
                        </View>
                    ) : null}

                    {errorMessage ? (
                        <View style={[styles.card, styles.errorCard]}>
                            <Text style={styles.errorTitle}>Request failed</Text>
                            <Text style={styles.errorText}>{errorMessage}</Text>
                        </View>
                    ) : null}

                    {result ? (
                        <View style={styles.card}>
                            <Text style={styles.label}>Detection result</Text>
                            <View style={styles.resultHeader}>
                                <Text style={styles.verdict}>{result.verdict}</Text>
                                <View style={styles.scoreChip}>
                                    <Text style={styles.scoreChipText}>
                                        {(result.ensemble_score * 100).toFixed(1)}%
                                    </Text>
                                </View>
                            </View>
                            <Text style={styles.resultMeta}>
                                Source: {result.filename || selectedAsset?.fileName || "image"}
                            </Text>

                            <View style={styles.metricList}>
                                <MetricRow
                                    label="FFT"
                                    value={result.detectors?.fft?.score}
                                />
                                <MetricRow
                                    label="ELA"
                                    value={result.detectors?.ela?.score}
                                />
                                <MetricRow
                                    label="Statistics"
                                    value={result.detectors?.statistics?.score}
                                />
                                <MetricRow
                                    label="Texture"
                                    value={result.detectors?.texture?.score}
                                />
                            </View>
                        </View>
                    ) : null}
                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

function MetricRow({ label, value }) {
    const safeValue = typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "--";
    return (
        <View style={styles.metricRow}>
            <Text style={styles.metricLabel}>{label}</Text>
            <Text style={styles.metricValue}>{safeValue}</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    safeArea: {
        flex: 1,
        backgroundColor: "#08171b",
    },
    flex: {
        flex: 1,
    },
    scrollContent: {
        padding: 20,
        gap: 16,
    },
    hero: {
        paddingTop: 8,
        paddingBottom: 6,
        gap: 8,
    },
    kicker: {
        color: "#72d8d2",
        fontSize: 13,
        fontWeight: "700",
        textTransform: "uppercase",
        letterSpacing: 1,
    },
    title: {
        color: "#f5fbfb",
        fontSize: 28,
        fontWeight: "700",
    },
    subtitle: {
        color: "#b3c8cc",
        fontSize: 15,
        lineHeight: 22,
    },
    card: {
        backgroundColor: "#102227",
        borderColor: "#1d373d",
        borderWidth: 1,
        borderRadius: 18,
        padding: 16,
        gap: 12,
    },
    label: {
        color: "#f5fbfb",
        fontSize: 16,
        fontWeight: "600",
    },
    input: {
        borderRadius: 12,
        borderWidth: 1,
        borderColor: "#29434a",
        backgroundColor: "#0a1a1e",
        color: "#f5fbfb",
        paddingHorizontal: 14,
        paddingVertical: 12,
        fontSize: 15,
    },
    helpText: {
        color: "#9cb3b8",
        fontSize: 13,
        lineHeight: 18,
    },
    emptyState: {
        color: "#9cb3b8",
        fontSize: 14,
    },
    previewWrap: {
        gap: 10,
    },
    preview: {
        width: "100%",
        height: 260,
        borderRadius: 14,
        backgroundColor: "#0a1a1e",
    },
    previewMeta: {
        color: "#cfe0e2",
        fontSize: 13,
    },
    buttonRow: {
        flexDirection: "row",
        gap: 12,
    },
    primaryButton: {
        flex: 1,
        backgroundColor: "#0b8f8a",
        borderRadius: 12,
        paddingVertical: 14,
        alignItems: "center",
    },
    secondaryButton: {
        flex: 1,
        backgroundColor: "#1b3237",
        borderRadius: 12,
        paddingVertical: 14,
        alignItems: "center",
    },
    primaryButtonText: {
        color: "#f5fbfb",
        fontSize: 15,
        fontWeight: "700",
    },
    secondaryButtonText: {
        color: "#d8e6e8",
        fontSize: 15,
        fontWeight: "700",
    },
    buttonDisabled: {
        opacity: 0.55,
    },
    loadingRow: {
        flexDirection: "row",
        alignItems: "center",
        gap: 10,
    },
    loadingText: {
        color: "#cfe0e2",
        fontSize: 14,
        flex: 1,
    },
    errorCard: {
        borderColor: "#6a2431",
        backgroundColor: "#271118",
    },
    errorTitle: {
        color: "#ffccd6",
        fontSize: 16,
        fontWeight: "700",
    },
    errorText: {
        color: "#ffb3c2",
        fontSize: 14,
        lineHeight: 20,
    },
    resultHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 12,
    },
    verdict: {
        color: "#f5fbfb",
        fontSize: 24,
        fontWeight: "700",
        flex: 1,
    },
    scoreChip: {
        backgroundColor: "#17373c",
        borderRadius: 999,
        paddingHorizontal: 12,
        paddingVertical: 8,
    },
    scoreChipText: {
        color: "#72d8d2",
        fontSize: 14,
        fontWeight: "700",
    },
    resultMeta: {
        color: "#9cb3b8",
        fontSize: 13,
    },
    metricList: {
        marginTop: 4,
        gap: 10,
    },
    metricRow: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        paddingVertical: 10,
        borderBottomWidth: 1,
        borderBottomColor: "#1c3439",
    },
    metricLabel: {
        color: "#d8e6e8",
        fontSize: 14,
    },
    metricValue: {
        color: "#f5fbfb",
        fontSize: 14,
        fontWeight: "700",
    },
});
