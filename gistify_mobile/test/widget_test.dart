import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:gistify/main.dart';

void main() {
  testWidgets('renders login screen by default', (WidgetTester tester) async {
    await tester.pumpWidget(const GistifyApp());

    expect(find.text('Welcome Back'), findsOneWidget);
    expect(find.byType(TextFormField), findsNWidgets(2));
    expect(find.widgetWithText(ElevatedButton, 'Log In'), findsOneWidget);
  });
}
