import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: RegexPage(),
    );
  }
}

class RegexPage extends StatefulWidget {
  const RegexPage({super.key});

  @override
  State<RegexPage> createState() => _RegexPageState();
}

class _RegexPageState extends State<RegexPage> {
  final TextEditingController textController = TextEditingController();
  final TextEditingController regexController = TextEditingController();
  List<String> matches = [];
  String? error;

  bool loading = false;

  // Future<void> submit() async {
  //   setState(() {
  //     loading = true;
  //     error = null;
  //     matches = [];
  //   });

  //   try {
  //     final response = await http.post(
  //       Uri.parse('http://10.0.2.2:5000/match'),
  //       headers: {'Content-Type': 'application/json'},
  //       body: jsonEncode({
  //         'test_string': textController.text,
  //         'pattern': regexController.text,
  //       }),
  //     );

  //     final data = jsonDecode(response.body);

  //     if (response.statusCode == 200) {
  //       setState(() {
  //         matches = List<String>.from(data['matches']);
  //       });
  //     } else {
  //       setState(() {
  //         error = data['error'];
  //       });
  //     }
  //   } catch (e) {
  //     setState(() {
  //       error = 'Server not reachable';
  //     });
  //   } finally {
  //     setState(() {
  //       loading = false;
  //     });
  //   }
  // }
  Future<void> submit() async {
    setState(() {
      loading = true;
      error = null;
      matches = [];
    });

    try {
      // Debug: show what we are sending
      print("Sending request to Flask...");
      print("Test string: ${textController.text}");
      print("Pattern: ${regexController.text}");

      final response = await http.post(
        // Uri.parse('http://10.0.2.2:5000/match'),
        Uri.parse('http://127.0.0.1:5000/match'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'test_string': textController.text,
          'pattern': regexController.text,
        }),
      );

      // Debug: show response status
      print("Response status: ${response.statusCode}");
      print("Response body: ${response.body}");

      final data = jsonDecode(response.body);

      if (response.statusCode == 200) {
        setState(() {
          matches = List<String>.from(data['matches']);
        });
      } else {
        setState(() {
          error = data['error'];
        });
      }
    } catch (e) {
      // Debug: show actual error
      print("Error caught: $e");
      setState(() {
        error = 'Server not reachable';
      });
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Regex Matcher')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: textController,
              maxLines: 4,
              decoration: const InputDecoration(
                labelText: 'Test String',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: regexController,
              decoration: const InputDecoration(
                labelText: 'Regex Pattern',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: loading ? null : submit,
              child: loading
                  ? const CircularProgressIndicator()
                  : const Text('Match'),
            ),
            const SizedBox(height: 16),
            if (error != null)
              Text(error!, style: const TextStyle(color: Colors.red)),
            if (matches.isNotEmpty)
              Expanded(
                child: ListView.builder(
                  itemCount: matches.length,
                  itemBuilder: (context, index) {
                    return ListTile(title: Text(matches[index]));
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// }
